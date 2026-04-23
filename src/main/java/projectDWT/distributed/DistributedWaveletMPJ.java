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

        Logger.info("[Rank " + rank + "/" + size + "] started - TRANSPOSE APPROACH (EXACT MATCH sequential)");

        double thresholdPercent = 5.0;
        BufferedImage originalImage = null;
        int height = 0, width = 0;

        if (isRoot) {
            final String[] chosen = {null, "5"};
            SwingUtilities.invokeAndWait(() -> {
                JFileChooser fc = new JFileChooser();
                fc.setDialogTitle("Load Image — Distributed DWT");
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

        int[] dims = new int[]{height, width};
        MPI.COMM_WORLD.Bcast(dims, 0, 2, MPI.INT, 0);
        height = dims[0];
        width = dims[1];

        double[] thr = {thresholdPercent};
        MPI.COMM_WORLD.Bcast(thr, 0, 1, MPI.DOUBLE, 0);
        thresholdPercent = thr[0];
        int maxLevels = -1;

        //partitioning
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

        double[] localData = new double[localRows * width * 3];

        double[] flatData = new double[height * width * 3];
        if (isRoot) {
            for (int y = 0; y < height; y++) {
                for (int x = 0; x < width; x++) {
                    int rgb = originalImage.getRGB(x, y);
                    int idx = (y * width + x) * 3;
                    flatData[idx]     = (rgb >> 16) & 0xFF;
                    flatData[idx + 1] = (rgb >> 8) & 0xFF;
                    flatData[idx + 2] = rgb & 0xFF;
                }
            }
        }

        MPI.COMM_WORLD.Scatterv(flatData, 0, sendCounts, displs, MPI.DOUBLE,
                localData, 0, localRows * width * 3, MPI.DOUBLE, 0);

        TimerMs timer = new TimerMs();
        if (rank == 0) timer.start();

        HaarWavelet2D haar = new HaarWavelet2D();

        double[][] r = extractChannel(localData, localRows, width, 0);
        double[][] g = extractChannel(localData, localRows, width, 1);
        double[][] b = extractChannel(localData, localRows, width, 2);

        haar.horizontalForwardInPlace(r, maxLevels);
        haar.horizontalForwardInPlace(g, maxLevels);
        haar.horizontalForwardInPlace(b, maxLevels);

        double[][] fullR = allGatherMatrix(r, height, width, rank, size);
        double[][] fullG = allGatherMatrix(g, height, width, rank, size);
        double[][] fullB = allGatherMatrix(b, height, width, rank, size);

        haar.verticalForwardInPlace(fullR, maxLevels);
        haar.verticalForwardInPlace(fullG, maxLevels);
        haar.verticalForwardInPlace(fullB, maxLevels);

        //Threshold
        double maxAbs = Math.max(MatrixUtil.maxAbs(fullR),
                Math.max(MatrixUtil.maxAbs(fullG), MatrixUtil.maxAbs(fullB)));

        double p = thresholdPercent / 100.0;
        double threshold = p * p * maxAbs;

        MatrixUtil.thresholdInPlace(fullR, threshold);
        MatrixUtil.thresholdInPlace(fullG, threshold);
        MatrixUtil.thresholdInPlace(fullB, threshold);

        haar.verticalInverseInPlace(fullR, maxLevels);
        haar.verticalInverseInPlace(fullG, maxLevels);
        haar.verticalInverseInPlace(fullB, maxLevels);

        int startRow = calculateStartRow(rank, height, size);
        double[][] localR = extractLocalStrip(fullR, startRow, localRows);
        double[][] localG = extractLocalStrip(fullG, startRow, localRows);
        double[][] localB = extractLocalStrip(fullB, startRow, localRows);

        haar.horizontalInverseInPlace(localR, maxLevels);
        haar.horizontalInverseInPlace(localG, maxLevels);
        haar.horizontalInverseInPlace(localB, maxLevels);

        double[] resultLocal = combineChannels(localR, localG, localB, localRows, width);

        double[] finalResult = (rank == 0) ? new double[height * width * 3] : null;
        MPI.COMM_WORLD.Gatherv(resultLocal, 0, resultLocal.length, MPI.DOUBLE,
                finalResult, 0, sendCounts, displs, MPI.DOUBLE, 0);

        if (rank == 0) {
            long elapsed = timer.stop();
            Logger.info("Distributed DWT (Transpose + EXACT MATCH) finished in " + elapsed + " ms");

            double[][][] rgb = toRGBMatrix(finalResult, height, width);
            BufferedImage reconstructed = ImageRGB.toImage(rgb[0], rgb[1], rgb[2]);

            showResultWindow(originalImage, reconstructed, elapsed);
        }
    }

    //helper functions
    private static int calculateStartRow(int rank, int height, int size) {
        int rowsPerProc = height / size;
        int remainder = height % size;
        int start = 0;
        for (int i = 0; i < rank; i++) {
            start += rowsPerProc + (i < remainder ? 1 : 0);
        }
        return start;
    }

    private static double[][] allGatherMatrix(double[][] local, int height, int width, int rank, int size) {
        double[] flatLocal = flatten(local);
        int localElems = local.length * width;

        int[] recvCounts = new int[size];
        int[] displs = new int[size];
        for (int i = 0; i < size; i++) {
            int rows = height / size + (i < height % size ? 1 : 0);
            recvCounts[i] = rows * width;
            displs[i] = (i == 0) ? 0 : displs[i-1] + recvCounts[i-1];
        }

        double[] fullFlat = new double[height * width];
        MPI.COMM_WORLD.Allgatherv(flatLocal, 0, localElems, MPI.DOUBLE, fullFlat, 0, recvCounts, displs, MPI.DOUBLE);
        return unflatten(fullFlat, height, width);
    }

    private static double[][] extractLocalStrip(double[][] full, int startRow, int localRows) {
        double[][] strip = new double[localRows][full[0].length];
        for (int i = 0; i < localRows; i++) {
            System.arraycopy(full[startRow + i], 0, strip[i], 0, full[0].length);
        }
        return strip;
    }

    private static double[] flatten(double[][] mat) {
        int h = mat.length, w = mat[0].length;
        double[] flat = new double[h * w];
        for (int i = 0; i < h; i++) System.arraycopy(mat[i], 0, flat, i * w, w);
        return flat;
    }

    private static double[][] unflatten(double[] flat, int h, int w) {
        double[][] mat = new double[h][w];
        for (int i = 0; i < h; i++) System.arraycopy(flat, i * w, mat[i], 0, w);
        return mat;
    }

    private static double[][] extractChannel(double[] flat, int rows, int width, int channel) {
        double[][] mat = new double[rows][width];
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < width; j++)
                mat[i][j] = flat[(i * width + j) * 3 + channel];
        return mat;
    }

    private static double[] combineChannels(double[][] r, double[][] g, double[][] b, int rows, int width) {
        double[] flat = new double[rows * width * 3];
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

    private static double[][][] toRGBMatrix(double[] flat, int h, int w) {
        double[][][] rgb = new double[3][h][w];
        for (int i = 0; i < h; i++)
            for (int j = 0; j < w; j++)
                for (int c = 0; c < 3; c++)
                    rgb[c][i][j] = flat[(i * w + j) * 3 + c];
        return rgb;
    }

    private static void showResultWindow(BufferedImage original, BufferedImage reconstructed, long timeMs) {
        SwingUtilities.invokeLater(() -> {
            JFrame f = new JFrame("Distributed DWT – Time: " + timeMs + " ms");
            f.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
            f.setLayout(new BorderLayout());

            JPanel panel = new JPanel(new GridLayout(1, 2));
            JLabel origLabel = new JLabel(new ImageIcon(scaleImage(original)));
            JLabel recLabel = new JLabel(new ImageIcon(scaleImage(reconstructed)));

            origLabel.setBorder(BorderFactory.createTitledBorder("Original"));
            recLabel.setBorder(BorderFactory.createTitledBorder("Reconstructed (Distributed)"));

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
            f.setSize(1150, 720);
            f.setLocationRelativeTo(null);
            f.setVisible(true);
        });
    }

    private static Image scaleImage(BufferedImage img) {
        int target = 520;
        double aspect = (double) img.getWidth() / img.getHeight();
        int w = target;
        int h = (int) (w / aspect);
        if (h > target) { h = target; w = (int)(h * aspect); }
        return img.getScaledInstance(w, h, Image.SCALE_SMOOTH);
    }

    private static double clampThreshold(String s) {
        try { return Math.max(0, Math.min(100, Double.parseDouble(s))); }
        catch (Exception e) { return 5.0; }
    }
}