package projectDWT.core;

import util.Logger;

import java.awt.image.BufferedImage;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;

public class ParallelWaveletPipeline {

    private final HaarWavelet2D haar = new HaarWavelet2D();

    public BufferedImage compressAndReconstruct(BufferedImage img, int thresholdPercent) throws InterruptedException {
        ImageRGB rgb = ImageRGB.fromImage(img);
        final double[][] r = rgb.r;
        final double[][] g = rgb.g;
        final double[][] b = rgb.b;

        final double[][] rCoef = new double[r.length][r[0].length];
        final double[][] gCoef = new double[g.length][g[0].length];
        final double[][] bCoef = new double[b.length][b[0].length];

        final double[][] rRec = new double[r.length][r[0].length];
        final double[][] gRec = new double[g.length][g[0].length];
        final double[][] bRec = new double[b.length][b[0].length];

        int numThreads = Runtime.getRuntime().availableProcessors();
        ExecutorService executor = Executors.newFixedThreadPool(numThreads);

        try {
            //forward DWT by channels
            executor.submit(() -> copy2D(haar.forward(r, -1), rCoef));
            executor.submit(() -> copy2D(haar.forward(g, -1), gCoef));
            executor.submit(() -> copy2D(haar.forward(b, -1), bCoef));

            //waiting
            executor.shutdown();
            if (!executor.awaitTermination(60, TimeUnit.SECONDS)) {
                executor.shutdownNow();
                throw new InterruptedException("Forward DWT timeout");
            }

            //threshold formula
            double maxAbs = Math.max(MatrixUtil.maxAbs(rCoef),
                    Math.max(MatrixUtil.maxAbs(gCoef), MatrixUtil.maxAbs(bCoef)));
            double p = thresholdPercent / 100.0;
            double threshold = p * p * maxAbs;

            Logger.info("Max abs coefficient: " + maxAbs + " | Applied threshold: " + threshold);

            executor = Executors.newFixedThreadPool(numThreads);

            executor.submit(() -> parallelThresholdInPlace(rCoef, threshold));
            executor.submit(() -> parallelThresholdInPlace(gCoef, threshold));
            executor.submit(() -> parallelThresholdInPlace(bCoef, threshold));

            executor.shutdown();
            if (!executor.awaitTermination(30, TimeUnit.SECONDS)) {
                executor.shutdownNow();
                throw new InterruptedException("Thresholding timeout");
            }

            //inverse DWT
            executor = Executors.newFixedThreadPool(numThreads);

            executor.submit(() -> copy2D(haar.inverse(rCoef, -1), rRec));
            executor.submit(() -> copy2D(haar.inverse(gCoef, -1), gRec));
            executor.submit(() -> copy2D(haar.inverse(bCoef, -1), bRec));

            executor.shutdown();
            if (!executor.awaitTermination(60, TimeUnit.SECONDS)) {
                executor.shutdownNow();
                throw new InterruptedException("Inverse DWT timeout");
            }

        } catch (InterruptedException e) {
            Logger.error("Parallel processing was interrupted: " + e.getMessage());
            throw e;
        } catch (Exception e) {
            Logger.error("Error in parallel processing: " + e.getMessage());
            throw new RuntimeException(e);
        } finally {
            if (executor != null && !executor.isShutdown()) {
                executor.shutdownNow();
            }
        }

        return ImageRGB.toImage(rRec, gRec, bRec);
    }

    private static void copy2D(double[][] src, double[][] dest) {
        for (int i = 0; i < src.length; i++) {
            System.arraycopy(src[i], 0, dest[i], 0, src[i].length);
        }
    }

    private void parallelThresholdInPlace(double[][] coef, double threshold) {
        int height = coef.length;
        int width = coef[0].length;
        int numThreads = Runtime.getRuntime().availableProcessors();
        int chunk = (height + numThreads - 1) / numThreads;

        ExecutorService threshExec = Executors.newFixedThreadPool(numThreads);

        try {
            for (int i = 0; i < numThreads; i++) {
                final int startY = i * chunk;
                final int endY = Math.min(startY + chunk, height);
                if (startY >= endY) break;

                threshExec.submit(() -> {
                    for (int y = startY; y < endY; y++) {
                        for (int x = 0; x < width; x++) {
                            if (Math.abs(coef[y][x]) < threshold) {
                                coef[y][x] = 0.0;
                            }
                        }
                    }
                });
            }

            threshExec.shutdown();
            threshExec.awaitTermination(20, TimeUnit.SECONDS);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            Logger.error("Thresholding thread interrupted");
        } finally {
            if (threshExec != null && !threshExec.isShutdown()) {
                threshExec.shutdownNow();
            }
        }
    }
}