package projectDWT.distributed;

import mpi.MPI;
import projectDWT.core.HaarWavelet2D;
import projectDWT.core.ImageRGB;
import projectDWT.core.MatrixUtil;
import util.Logger;

import javax.imageio.ImageIO;
import javax.swing.*;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;

public class DistributedWaveletMPJ {

    public static void runMPI(String[] args, boolean isRoot) throws Exception {
        int rank = MPI.COMM_WORLD.Rank();
        int size = MPI.COMM_WORLD.Size();

        Logger.info("[Rank " + rank + "/" + size + "] Process started");

        float thresholdPercent = 5.0f;
        BufferedImage originalImage = null;
        int height = 0, width = 0;

        if (isRoot) {
            final String[] chosen = {null, "5"};
            SwingUtilities.invokeAndWait(() -> {
                JFileChooser fc = new JFileChooser();
                fc.setDialogTitle("Load Image — Distributed DWT");
                if (fc.showOpenDialog(null) == JFileChooser.APPROVE_OPTION) {
                    chosen[0] = fc.getSelectedFile().getAbsolutePath();
                }
                String t = JOptionPane.showInputDialog(null,
                        "Threshold (0–100%)", "5");
                if (t != null && !t.isBlank()) {
                    chosen[1] = t;
                }
            });

            if (chosen[0] != null) {
                thresholdPercent = clampThreshold(chosen[1]);
                originalImage = ImageIO.read(new File(chosen[0]));
                height = originalImage.getHeight();
                width = originalImage.getWidth();
            }
        }

        // Broadcasting image dimensions
        int[] dims = {height, width};
        MPI.COMM_WORLD.Bcast(dims, 0, 2, MPI.INT, 0);
        height = dims[0];
        width = dims[1];

        if (height == 0 || width == 0) {
            Logger.info("[Rank " + rank + "] User cancelled — exiting.");
            return;
        }

        // broadcast threshold
        float[] thrArr = {thresholdPercent};
        MPI.COMM_WORLD.Bcast(thrArr, 0, 1, MPI.FLOAT, 0);
        thresholdPercent = thrArr[0];

        //row partition for original image
        int[] rowCounts = partitionCounts(height, size);
        int[] rowDispls = prefixDispls(rowCounts);

        //column partition used for transpose distribution
        int[] colCounts = partitionCounts(width, size);
        int[] colDispls = prefixDispls(colCounts);

        int localRows = rowCounts[rank];
        int localColsAfterTranspose = colCounts[rank];

        //root packs original image into one interleaved RGB buffer
        float[] flatFull = null;
        if (isRoot) {
            flatFull = new float[height * width * 3];
            ImageRGB rgb = ImageRGB.fromImage(originalImage);
            packRGB(rgb.r, rgb.g, rgb.b, flatFull, height, width);
        }

        //scattering original rows to all ranks
        float[] localRGB = new float[localRows * width * 3];
        MPI.COMM_WORLD.Scatterv(
                flatFull, 0, multiplyCounts(rowCounts, width * 3), multiplyDispls(rowDispls, width * 3), MPI.FLOAT,
                localRGB, 0, localRows * width * 3, MPI.FLOAT, 0
        );
        flatFull = null;

        //unpack to separate channels
        float[][] localR = new float[localRows][width];
        float[][] localG = new float[localRows][width];
        float[][] localB = new float[localRows][width];
        unpackRGB(localRGB, localR, localG, localB, localRows, width);
        localRGB = null;

        HaarWavelet2D haar = new HaarWavelet2D();

        long totalStart = (rank == 0) ? System.currentTimeMillis() : 0;
        long compMs = 0;
        long commMs = 0;

        //horizontal forward DWT on row-distributed data
        long t0 = System.currentTimeMillis();
        if (localRows > 0) {
            haar.horizontalForwardInPlace(localR, -1);
            haar.horizontalForwardInPlace(localG, -1);
            haar.horizontalForwardInPlace(localB, -1);
        }
        compMs += System.currentTimeMillis() - t0;

        //distributed transpose: rows -> columns
        t0 = System.currentTimeMillis();
        localR = distributedTranspose(localR, height, rowCounts, rowDispls, colCounts, colDispls);
        localG = distributedTranspose(localG, height, rowCounts, rowDispls, colCounts, colDispls);
        localB = distributedTranspose(localB, height, rowCounts, rowDispls, colCounts, colDispls);
        commMs += System.currentTimeMillis() - t0;

        //horizontal forward DWT on transposed data = vertical DWT on original
        t0 = System.currentTimeMillis();
        if (localColsAfterTranspose > 0) {
            haar.horizontalForwardInPlace(localR, -1);
            haar.horizontalForwardInPlace(localG, -1);
            haar.horizontalForwardInPlace(localB, -1);
        }
        compMs += System.currentTimeMillis() - t0;

        //global max coefficient with allreduce
        float localMax = 0.0f;
        if (localR.length > 0) {
            localMax = Math.max(MatrixUtil.maxAbs(localR),
                    Math.max(MatrixUtil.maxAbs(localG), MatrixUtil.maxAbs(localB)));
        }

        float[] maxSend = {localMax};
        float[] maxRecv = {0.0f};
        MPI.COMM_WORLD.Allreduce(maxSend, 0, maxRecv, 0, 1, MPI.FLOAT, MPI.MAX);
        float globalMax = maxRecv[0];

        float threshold = (thresholdPercent / 100.0f) * (thresholdPercent / 100.0f) * globalMax;
        Logger.info("[Rank " + rank + "] MaxAbs=" + globalMax + " | Threshold=" + threshold);

        //threshold locally on every rank
        t0 = System.currentTimeMillis();
        thresholdInPlace(localR, threshold);
        thresholdInPlace(localG, threshold);
        thresholdInPlace(localB, threshold);
        compMs += System.currentTimeMillis() - t0;

        //inverse horizontal DWT on transposed data
        t0 = System.currentTimeMillis();
        if (localColsAfterTranspose > 0) {
            haar.horizontalInverseInPlace(localR, -1);
            haar.horizontalInverseInPlace(localG, -1);
            haar.horizontalInverseInPlace(localB, -1);
        }
        compMs += System.currentTimeMillis() - t0;

        //transpose back to original row distribution
        t0 = System.currentTimeMillis();
        localR = distributedTranspose(localR, width, colCounts, colDispls, rowCounts, rowDispls);
        localG = distributedTranspose(localG, width, colCounts, colDispls, rowCounts, rowDispls);
        localB = distributedTranspose(localB, width, colCounts, colDispls, rowCounts, rowDispls);
        commMs += System.currentTimeMillis() - t0;

        //final inverse horizontal DWT on original row distribution
        t0 = System.currentTimeMillis();
        if (localRows > 0) {
            haar.horizontalInverseInPlace(localR, -1);
            haar.horizontalInverseInPlace(localG, -1);
            haar.horizontalInverseInPlace(localB, -1);
        }
        compMs += System.currentTimeMillis() - t0;

        //Pack local result and gather to root
        float[] localFinal = new float[localRows * width * 3];
        packRGB(localR, localG, localB, localFinal, localRows, width);

        float[] finalFull = isRoot ? new float[height * width * 3] : null;

        t0 = System.currentTimeMillis();
        MPI.COMM_WORLD.Gatherv(
                localFinal, 0, localRows * width * 3, MPI.FLOAT,
                finalFull, 0,
                multiplyCounts(rowCounts, width * 3),
                multiplyDispls(rowDispls, width * 3),
                MPI.FLOAT, 0
        );
        commMs += System.currentTimeMillis() - t0;

        if (rank == 0) {
            long totalMs = System.currentTimeMillis() - totalStart;

            Logger.info("=== DISTRIBUTED DWT FINISHED ===");
            Logger.info("Total time         : " + totalMs + " ms");
            Logger.info("Computation time   : " + compMs + " ms");
            Logger.info("Communication time : " + commMs + " ms");

            float[][] outR = new float[height][width];
            float[][] outG = new float[height][width];
            float[][] outB = new float[height][width];
            unpackRGB(finalFull, outR, outG, outB, height, width);

            BufferedImage reconstructed = ImageRGB.toImage(outR, outG, outB);
            showResultWindow(originalImage, reconstructed, totalMs, compMs, commMs, size);
        }
    }

    private static float[][] distributedTranspose(float[][] local, int globalRows, int[] srcRowCounts, int[] srcRowDispls,
                                                  int[] dstRowCounts, int[] dstRowDispls) throws Exception {
        int rank = MPI.COMM_WORLD.Rank();
        int size = MPI.COMM_WORLD.Size();

        int srcRows = srcRowCounts[rank];
        int dstRows = dstRowCounts[rank];

        //send
        int[] sendCounts = new int[size];
        int[] sendDispls = new int[size];
        int sendOff = 0;
        for (int d = 0; d < size; d++) {
            sendCounts[d] = srcRows * dstRowCounts[d];
            sendDispls[d] = sendOff;
            sendOff += sendCounts[d];
        }

        //Receive from each source
        int[] recvCounts = new int[size];
        int[] recvDispls = new int[size];
        int recvOff = 0;
        for (int s = 0; s < size; s++) {
            recvCounts[s] = srcRowCounts[s] * dstRows;
            recvDispls[s] = recvOff;
            recvOff += recvCounts[s];
        }

        float[] sendBuf = new float[sendOff];
        int p = 0;

        //Pack in source-row-major order then destination-column slice
        if (local.length > 0 && local[0].length > 0) {
            for (int d = 0; d < size; d++) {
                int colStart = dstRowDispls[d];
                int colEnd = colStart + dstRowCounts[d];
                for (int y = 0; y < srcRows; y++) {
                    for (int x = colStart; x < colEnd; x++) {
                        sendBuf[p++] = local[y][x];
                    }
                }
            }
        }

        float[] recvBuf = new float[recvOff];

        MPI.COMM_WORLD.Alltoallv(
                sendBuf, 0, sendCounts, sendDispls, MPI.FLOAT,
                recvBuf, 0, recvCounts, recvDispls, MPI.FLOAT
        );

        //reconstructing transposed local matrix:
        //output shape = [dstRows][globalRows]
        float[][] out = new float[dstRows][globalRows];

        int ptr = 0;
        for (int s = 0; s < size; s++) {
            int srcRowsCount = srcRowCounts[s];
            int rowBase = srcRowDispls[s];

            for (int y = 0; y < srcRowsCount; y++) {
                int globalRowIndex = rowBase + y;
                for (int x = 0; x < dstRows; x++) {
                    out[x][globalRowIndex] = recvBuf[ptr++];
                }
            }
        }

        return out;
    }

    private static void packRGB(float[][] r, float[][] g, float[][] b,
                                float[] out, int rows, int width) {
        for (int y = 0; y < rows; y++) {
            int base = y * width;
            for (int x = 0; x < width; x++) {
                int idx = (base + x) * 3;
                out[idx] = r[y][x];
                out[idx + 1] = g[y][x];
                out[idx + 2] = b[y][x];
            }
        }
    }

    private static void unpackRGB(float[] in, float[][] r, float[][] g, float[][] b,
                                  int rows, int width) {
        for (int y = 0; y < rows; y++) {
            int base = y * width;
            for (int x = 0; x < width; x++) {
                int idx = (base + x) * 3;
                r[y][x] = in[idx];
                g[y][x] = in[idx + 1];
                b[y][x] = in[idx + 2];
            }
        }
    }

    private static void thresholdInPlace(float[][] m, float thr) {
        if (m.length == 0) return;
        int h = m.length;
        int w = m[0].length;
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                if (Math.abs(m[y][x]) < thr) {
                    m[y][x] = 0.0f;
                }
            }
        }
    }

    private static int[] partitionCounts(int total, int size) {
        int[] counts = new int[size];
        int base = total / size;
        int rem = total % size;
        for (int i = 0; i < size; i++) {
            counts[i] = base + (i < rem ? 1 : 0);
        }
        return counts;
    }

    private static int[] prefixDispls(int[] counts) {
        int[] displs = new int[counts.length];
        int off = 0;
        for (int i = 0; i < counts.length; i++) {
            displs[i] = off;
            off += counts[i];
        }
        return displs;
    }

    private static int[] multiplyCounts(int[] counts, int factor) {
        int[] out = new int[counts.length];
        for (int i = 0; i < counts.length; i++) {
            out[i] = counts[i] * factor;
        }
        return out;
    }

    private static int[] multiplyDispls(int[] displs, int factor) {
        int[] out = new int[displs.length];
        for (int i = 0; i < displs.length; i++) {
            out[i] = displs[i] * factor;
        }
        return out;
    }

    private static void showResultWindow(BufferedImage original,
                                         BufferedImage reconstructed,
                                         long totalMs, long compMs,
                                         long commMs, int numProcs) {
        SwingUtilities.invokeLater(() -> {
            JFrame f = new JFrame(
                    "Distributed DWT [" + numProcs + " proc] — Total: "
                            + totalMs + " ms | Comp: " + compMs
                            + " ms | Comm: " + commMs + " ms");
            f.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
            f.setLayout(new BorderLayout());

            JPanel panel = new JPanel(new GridLayout(1, 2));
            JLabel origLabel = new JLabel(new ImageIcon(scaleImage(original)));
            JLabel recLabel = new JLabel(new ImageIcon(scaleImage(reconstructed)));
            origLabel.setBorder(BorderFactory.createTitledBorder("Original"));
            recLabel.setBorder(BorderFactory.createTitledBorder(
                    "Reconstructed — " + numProcs + " processes"));
            panel.add(new JScrollPane(origLabel));
            panel.add(new JScrollPane(recLabel));

            JButton saveBtn = new JButton("Save reconstructed image");
            saveBtn.addActionListener(e -> {
                JFileChooser chooser = new JFileChooser();
                if (chooser.showSaveDialog(f) == JFileChooser.APPROVE_OPTION) {
                    try {
                        ImageIO.write(reconstructed, "png", chooser.getSelectedFile());
                        Logger.info("Image saved.");
                    } catch (Exception ex) {
                        Logger.error("Save failed: " + ex.getMessage());
                    }
                }
            });

            JLabel statsLabel = new JLabel(
                    "  Processes: " + numProcs
                            + "   |   Total: " + totalMs + " ms"
                            + "   |   Computation: " + compMs + " ms"
                            + "   |   Communication: " + commMs + " ms",
                    SwingConstants.CENTER);

            JPanel south = new JPanel(new BorderLayout());
            south.add(saveBtn, BorderLayout.WEST);
            south.add(statsLabel, BorderLayout.CENTER);

            f.add(panel, BorderLayout.CENTER);
            f.add(south, BorderLayout.SOUTH);
            f.setSize(1200, 760);
            f.setLocationRelativeTo(null);
            f.setVisible(true);
        });
    }

    private static Image scaleImage(BufferedImage img) {
        int target = 560;
        float aspect = (float) img.getWidth() / img.getHeight();
        int w = target;
        int h = (int) (w / aspect);
        if (h > target) {
            h = target;
            w = (int) (h * aspect);
        }
        return img.getScaledInstance(w, h, Image.SCALE_SMOOTH);
    }

    private static float clampThreshold(String s) {
        try {
            return Math.max(0f, Math.min(100f, Float.parseFloat(s)));
        } catch (Exception e) {
            return 5.0f;
        }
    }
}