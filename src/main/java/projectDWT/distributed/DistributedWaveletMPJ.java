package projectDWT.distributed;

import mpi.*;
import projectDWT.core.*;
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

        Logger.info("[Rank " + rank + "/" + size + "] HYBRID OPTIMIZED (Horizontal MPI + Vertical on Root)");

        float thresholdPercent = 5.0f;
        BufferedImage originalImage = null;
        int height = 0, width = 0;

        if (isRoot) {
            final String[] chosen = {null, "5"};
            SwingUtilities.invokeAndWait(() -> {
                JFileChooser fc = new JFileChooser();
                fc.setDialogTitle("Load Image — Distributed DWT (Hybrid)");
                if (fc.showOpenDialog(null) == JFileChooser.APPROVE_OPTION)
                    chosen[0] = fc.getSelectedFile().getAbsolutePath();

                String t = JOptionPane.showInputDialog(null, "Threshold (0–100%)", "5");
                if (t != null && !t.isBlank()) chosen[1] = t;
            });

            if (chosen[0] == null) return;

            thresholdPercent = clampThreshold(chosen[1]);
            originalImage = ImageIO.read(new File(chosen[0]));
            height = originalImage.getHeight();
            width = originalImage.getWidth();
        }

        //broadcast
        int[] dims = new int[]{height, width};
        MPI.COMM_WORLD.Bcast(dims, 0, 2, MPI.INT, 0);
        height = dims[0];
        width = dims[1];

        float[] thr = {thresholdPercent};
        MPI.COMM_WORLD.Bcast(thr, 0, 1, MPI.FLOAT, 0);
        thresholdPercent = thr[0];

        int maxLevels = -1;

        //row partitioning
        int rowsPerProc = height / size;
        int remainder = height % size;
        int[] sendCounts = new int[size];
        int[] displs = new int[size];
        int offset = 0;
        for (int i = 0; i < size; i++) {
            int rows = rowsPerProc + (i < remainder ? 1 : 0);
            sendCounts[i] = rows * width * 3;
            displs[i] = offset;
            offset += sendCounts[i];
        }

        int localRows = rowsPerProc + (rank < remainder ? 1 : 0);
        float[] localData = new float[localRows * width * 3];

        float[] flatData = new float[height * width * 3];
        if (isRoot) {
            ImageRGB rgb = ImageRGB.fromImage(originalImage);
            for (int y = 0; y < height; y++) {
                for (int x = 0; x < width; x++) {
                    int idx = (y * width + x) * 3;
                    flatData[idx]     = rgb.r[y][x];
                    flatData[idx + 1] = rgb.g[y][x];
                    flatData[idx + 2] = rgb.b[y][x];
                }
            }
        }

        MPI.COMM_WORLD.Scatterv(flatData, 0, sendCounts, displs, MPI.FLOAT,
                localData, 0, localRows * width * 3, MPI.FLOAT, 0);

        TimerMs totalTimer = new TimerMs();
        TimerMs compTimer = new TimerMs();
        TimerMs commTimer = new TimerMs();

        if (rank == 0) totalTimer.start();

        HaarWavelet2D haar = new HaarWavelet2D();

        float[][] localR = extractChannel(localData, localRows, width, 0);
        float[][] localG = extractChannel(localData, localRows, width, 1);
        float[][] localB = extractChannel(localData, localRows, width, 2);

        //horizontal forward
        compTimer.start();
        haar.horizontalForwardInPlace(localR, maxLevels);
        haar.horizontalForwardInPlace(localG, maxLevels);
        haar.horizontalForwardInPlace(localB, maxLevels);
        compTimer.stop();

        //gather to root
        commTimer.start();
        float[][] fullR = gatherToRoot(localR, height, width, rank, size);
        float[][] fullG = gatherToRoot(localG, height, width, rank, size);
        float[][] fullB = gatherToRoot(localB, height, width, rank, size);
        commTimer.stop();

        //vertical + threshold + inverse vertical only root
        if (rank == 0) {
            compTimer.start();

            haar.verticalForwardInPlace(fullR, maxLevels);
            haar.verticalForwardInPlace(fullG, maxLevels);
            haar.verticalForwardInPlace(fullB, maxLevels);

            float maxAbs = Math.max(MatrixUtil.maxAbs(fullR),
                    Math.max(MatrixUtil.maxAbs(fullG), MatrixUtil.maxAbs(fullB)));

            float threshold = (thresholdPercent / 100.0f) * (thresholdPercent / 100.0f) * maxAbs;
            Logger.info("[ROOT] Max abs: " + maxAbs + " | Threshold: " + threshold);

            MatrixUtil.thresholdInPlace(fullR, threshold);
            MatrixUtil.thresholdInPlace(fullG, threshold);
            MatrixUtil.thresholdInPlace(fullB, threshold);

            haar.verticalInverseInPlace(fullR, maxLevels);
            haar.verticalInverseInPlace(fullG, maxLevels);
            haar.verticalInverseInPlace(fullB, maxLevels);

            compTimer.stop();
        }

        //scatter
        commTimer.start();
        int startRow = calculateStartRow(rank, height, size);
        float[][] localRinv = scatterFromRoot(fullR, startRow, localRows, height, width, rank, size);
        float[][] localGinv = scatterFromRoot(fullG, startRow, localRows, height, width, rank, size);
        float[][] localBinv = scatterFromRoot(fullB, startRow, localRows, height, width, rank, size);
        commTimer.stop();

        //local horizontal inverse
        compTimer.start();
        haar.horizontalInverseInPlace(localRinv, maxLevels);
        haar.horizontalInverseInPlace(localGinv, maxLevels);
        haar.horizontalInverseInPlace(localBinv, maxLevels);
        compTimer.stop();

        float[] resultLocal = combineChannels(localRinv, localGinv, localBinv, localRows, width);

        float[] finalResult = (rank == 0) ? new float[height * width * 3] : null;

        commTimer.start();
        MPI.COMM_WORLD.Gatherv(resultLocal, 0, resultLocal.length, MPI.FLOAT,
                finalResult, 0, sendCounts, displs, MPI.FLOAT, 0);
        commTimer.stop();

        if (rank == 0) {
            long totalTime = totalTimer.stop();
            Logger.info("=== DISTRIBUTED HYBRID FINISHED ===");
            Logger.info("Total time          : " + totalTime + " ms");
            Logger.info("Computation time    : " + compTimer.stop() + " ms");
            Logger.info("Communication time  : " + commTimer.stop() + " ms");

            float[][][] rgb = toRGBMatrix(finalResult, height, width);
            BufferedImage reconstructed = ImageRGB.toImage(rgb[0], rgb[1], rgb[2]);

            showResultWindow(originalImage, reconstructed, totalTime);
        }
    }

    //helpers
    private static int calculateStartRow(int rank, int height, int size) {
        int rowsPerProc = height / size;
        int remainder = height % size;
        int start = 0;
        for (int i = 0; i < rank; i++) {
            start += rowsPerProc + (i < remainder ? 1 : 0);
        }
        return start;
    }

    private static float[][] gatherToRoot(float[][] local, int height, int width, int rank, int size) {
        float[] flatLocal = flatten(local);
        int localElems = local.length * width;

        int[] recvCounts = new int[size];
        int[] displs = new int[size];
        for (int i = 0; i < size; i++) {
            int rows = height / size + (i < height % size ? 1 : 0);
            recvCounts[i] = rows * width;
            displs[i] = (i == 0) ? 0 : displs[i - 1] + recvCounts[i - 1];
        }

        float[] fullFlat = (rank == 0) ? new float[height * width] : null;
        MPI.COMM_WORLD.Gatherv(flatLocal, 0, localElems, MPI.FLOAT, fullFlat, 0, recvCounts, displs, MPI.FLOAT, 0);

        return (rank == 0) ? unflatten(fullFlat, height, width) : null;
    }

    private static float[][] scatterFromRoot(float[][] full, int startRow, int localRows, int height, int width, int rank, int size) {
        int[] sendCounts = new int[size];
        int[] displs = new int[size];
        int offset = 0;
        for (int i = 0; i < size; i++) {
            int rows = height / size + (i < height % size ? 1 : 0);
            sendCounts[i] = rows * width;
            displs[i] = offset;
            offset += sendCounts[i];
        }

        float[] localFlat = new float[localRows * width];

        if (rank == 0) {
            float[] fullFlat = flatten(full);
            MPI.COMM_WORLD.Scatterv(fullFlat, 0, sendCounts, displs, MPI.FLOAT,
                    localFlat, 0, localRows * width, MPI.FLOAT, 0);
        } else {
            MPI.COMM_WORLD.Scatterv(null, 0, null, null, MPI.FLOAT,
                    localFlat, 0, localRows * width, MPI.FLOAT, 0);
        }
        return unflatten(localFlat, localRows, width);
    }

    private static float[] flatten(float[][] mat) {
        int h = mat.length, w = mat[0].length;
        float[] flat = new float[h * w];
        for (int i = 0; i < h; i++) {
            System.arraycopy(mat[i], 0, flat, i * w, w);
        }
        return flat;
    }

    private static float[][] unflatten(float[] flat, int h, int w) {
        float[][] mat = new float[h][w];
        for (int i = 0; i < h; i++) {
            System.arraycopy(flat, i * w, mat[i], 0, w);
        }
        return mat;
    }

    private static float[][] extractChannel(float[] flat, int rows, int width, int channel) {
        float[][] mat = new float[rows][width];
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < width; j++)
                mat[i][j] = flat[(i * width + j) * 3 + channel];
        return mat;
    }

    private static float[] combineChannels(float[][] r, float[][] g, float[][] b, int rows, int width) {
        float[] flat = new float[rows * width * 3];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < width; j++) {
                int idx = (i * width + j) * 3;
                flat[idx]     = r[i][j];
                flat[idx + 1] = g[i][j];
                flat[idx + 2] = b[i][j];
            }
        }
        return flat;
    }

    private static float[][][] toRGBMatrix(float[] flat, int h, int w) {
        float[][][] rgb = new float[3][h][w];
        for (int i = 0; i < h; i++)
            for (int j = 0; j < w; j++)
                for (int c = 0; c < 3; c++)
                    rgb[c][i][j] = flat[(i * w + j) * 3 + c];
        return rgb;
    }

    private static void showResultWindow(BufferedImage original, BufferedImage reconstructed, long timeMs) {
        SwingUtilities.invokeLater(() -> {
            JFrame f = new JFrame("Distributed DWT (Hybrid) – Time: " + timeMs + " ms");
            f.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
            f.setLayout(new BorderLayout());

            JPanel panel = new JPanel(new GridLayout(1, 2));
            JLabel origLabel = new JLabel(new ImageIcon(scaleImage(original)));
            JLabel recLabel = new JLabel(new ImageIcon(scaleImage(reconstructed)));

            origLabel.setBorder(BorderFactory.createTitledBorder("Original"));
            recLabel.setBorder(BorderFactory.createTitledBorder("Reconstructed (Distributed Hybrid)"));

            panel.add(new JScrollPane(origLabel));
            panel.add(new JScrollPane(recLabel));

            JButton saveBtn = new JButton("Save reconstructed image");
            saveBtn.addActionListener(e -> {
                JFileChooser chooser = new JFileChooser();
                if (chooser.showSaveDialog(f) == JFileChooser.APPROVE_OPTION) {
                    try { ImageIO.write(reconstructed, "png", chooser.getSelectedFile()); } catch (Exception ex) {}
                }
            });

            f.add(panel, BorderLayout.CENTER);
            f.add(saveBtn, BorderLayout.SOUTH);
            f.setSize(1200, 720);
            f.setLocationRelativeTo(null);
            f.setVisible(true);
        });
    }

    private static Image scaleImage(BufferedImage img) {
        int target = 560;
        float aspect = (float) img.getWidth() / img.getHeight();
        int w = target;
        int h = (int) (w / aspect);
        if (h > target) { h = target; w = (int)(h * aspect); }
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