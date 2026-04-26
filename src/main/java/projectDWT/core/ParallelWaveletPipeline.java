package projectDWT.core;

import util.Logger;

import java.awt.image.BufferedImage;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;

public class ParallelWaveletPipeline implements WaveletPipeline {

    private final HaarWavelet2D haar = new HaarWavelet2D();
    private ExecutorService executor;

    @Override
    public BufferedImage compressAndReconstruct(BufferedImage img, int thresholdPercent) throws Exception {
        ImageRGB rgb = ImageRGB.fromImage(img);
        final float[][] r = rgb.r;
        final float[][] g = rgb.g;
        final float[][] b = rgb.b;

        final float[][] rCoef = new float[r.length][r[0].length];
        final float[][] gCoef = new float[g.length][g[0].length];
        final float[][] bCoef = new float[b.length][b[0].length];

        final float[][] rRec = new float[r.length][r[0].length];
        final float[][] gRec = new float[g.length][g[0].length];
        final float[][] bRec = new float[b.length][b[0].length];

        int numThreads = Runtime.getRuntime().availableProcessors();
        executor = Executors.newFixedThreadPool(numThreads);

        try {
            //Forward DWT
            executor.submit(() -> copy2D(haar.forward(r, -1), rCoef));
            executor.submit(() -> copy2D(haar.forward(g, -1), gCoef));
            executor.submit(() -> copy2D(haar.forward(b, -1), bCoef));

            executor.shutdown();
            executor.awaitTermination(60, TimeUnit.SECONDS);

            //Threshold
            float maxAbs = Math.max(MatrixUtil.maxAbs(rCoef),
                    Math.max(MatrixUtil.maxAbs(gCoef), MatrixUtil.maxAbs(bCoef)));
            float thr = (thresholdPercent / 100.0f) * (thresholdPercent / 100.0f) * maxAbs;

            Logger.info("Max abs coefficient: " + maxAbs + " | Threshold: " + thr);

            executor = Executors.newFixedThreadPool(numThreads);
            executor.submit(() -> parallelThresholdInPlace(rCoef, thr));
            executor.submit(() -> parallelThresholdInPlace(gCoef, thr));
            executor.submit(() -> parallelThresholdInPlace(bCoef, thr));

            executor.shutdown();
            executor.awaitTermination(30, TimeUnit.SECONDS);

            //Inverse DWT
            executor = Executors.newFixedThreadPool(numThreads);
            executor.submit(() -> copy2D(haar.inverse(rCoef, -1), rRec));
            executor.submit(() -> copy2D(haar.inverse(gCoef, -1), gRec));
            executor.submit(() -> copy2D(haar.inverse(bCoef, -1), bRec));

            executor.shutdown();
            executor.awaitTermination(60, TimeUnit.SECONDS);

        } finally {
            if (executor != null && !executor.isShutdown()) {
                executor.shutdownNow();
            }
        }

        return ImageRGB.toImage(rRec, gRec, bRec);
    }

    private static void copy2D(float[][] src, float[][] dest) {
        for (int i = 0; i < src.length; i++) {
            System.arraycopy(src[i], 0, dest[i], 0, src[i].length);
        }
    }

    private void parallelThresholdInPlace(float[][] coef, float threshold) {
        int height = coef.length;
        int width = coef[0].length;
        int numThreads = Runtime.getRuntime().availableProcessors();
        int chunk = (height + numThreads - 1) / numThreads;

        ExecutorService threshExec = Executors.newFixedThreadPool(numThreads);
        try {
            for (int i = 0; i < numThreads; i++) {
                int startY = i * chunk;
                int endY = Math.min(startY + chunk, height);
                if (startY >= endY) break;

                threshExec.submit(() -> {
                    for (int y = startY; y < endY; y++) {
                        for (int x = 0; x < width; x++) {
                            if (Math.abs(coef[y][x]) < threshold) {
                                coef[y][x] = 0.0f;
                            }
                        }
                    }
                });
            }
            threshExec.shutdown();
            threshExec.awaitTermination(20, TimeUnit.SECONDS);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        } finally {
            if (threshExec != null && !threshExec.isShutdown()) threshExec.shutdownNow();
        }
    }
}