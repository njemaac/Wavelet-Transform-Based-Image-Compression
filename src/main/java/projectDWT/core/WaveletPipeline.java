package projectDWT.core;
import util.Logger;
import java.awt.image.BufferedImage;

public class WaveletPipeline {
    private final HaarWavelet2D haar = new HaarWavelet2D();

    public BufferedImage compressAndReconstruct(BufferedImage img, int thresholdPercent){
        ImageRGB rgb = ImageRGB.fromImage(img);

        Logger.debug("Forward DWT R...");
        float[][] rCoef = haar.forward(rgb.r, -1);
        Logger.debug("Forward DWT G...");
        float[][] gCoef = haar.forward(rgb.g, -1);
        Logger.debug("Forward DWT B...");
        float[][] bCoef = haar.forward(rgb.b, -1);

        float maxAbs = Math.max(MatrixUtil.maxAbs(rCoef),Math.max(MatrixUtil.maxAbs(gCoef),MatrixUtil.maxAbs(bCoef)));

        float p = (float) (thresholdPercent / 100.0);
        float thr = p * p * maxAbs;
        Logger.info("Max abs coefficient: " + maxAbs);
        Logger.info("Applying threshold: " + thresholdPercent + " -> value: " + thr);

        MatrixUtil.thresholdInPlace(rCoef, thr);
        MatrixUtil.thresholdInPlace(gCoef, thr);
        MatrixUtil.thresholdInPlace(bCoef, thr);

        //inverse
        Logger.debug("Inverse DWT R...");
        float[][] rRec = haar.inverse(rCoef, -1);
        Logger.debug("Inverse DWT G...");
        float[][] gRec = haar.inverse(gCoef, -1);
        Logger.debug("Inverse DWT B...");
        float[][] bRec = haar.inverse(bCoef, -1);

        return ImageRGB.toImage(rRec, gRec, bRec);
    }
}
