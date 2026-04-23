package projectDWT.core;
import util.Logger;
import java.awt.image.BufferedImage;

public class WaveletPipeline {
    private final HaarWavelet2D haar = new HaarWavelet2D();

    public BufferedImage compressAndReconstruct(BufferedImage img, int thresholdPercent){
        ImageRGB rgb = ImageRGB.fromImage(img);

        Logger.debug("Forward DWT R...");
        double[][] rCoef = haar.forward(rgb.r, -1);
        Logger.debug("Forward DWT G...");
        double[][] gCoef = haar.forward(rgb.g, -1);
        Logger.debug("Forward DWT B...");
        double[][] bCoef = haar.forward(rgb.b, -1);

        double maxAbs = Math.max(MatrixUtil.maxAbs(rCoef),Math.max(MatrixUtil.maxAbs(gCoef),MatrixUtil.maxAbs(bCoef)));

        double p = thresholdPercent / 100.0;
        double thr = p * p * maxAbs;
        Logger.info("Max abs coefficient: " + maxAbs);
        Logger.info("Applying threshold: " + thresholdPercent + " -> value: " + thr);

        MatrixUtil.thresholdInPlace(rCoef, thr);
        MatrixUtil.thresholdInPlace(gCoef, thr);
        MatrixUtil.thresholdInPlace(bCoef, thr);

        //inverse
        Logger.debug("Inverse DWT R...");
        double[][] rRec = haar.inverse(rCoef, -1);
        Logger.debug("Inverse DWT G...");
        double[][] gRec = haar.inverse(gCoef, -1);
        Logger.debug("Inverse DWT B...");
        double[][] bRec = haar.inverse(bCoef, -1);

        return ImageRGB.toImage(rRec, gRec, bRec);
    }
}
