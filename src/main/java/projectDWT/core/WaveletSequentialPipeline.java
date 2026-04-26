package projectDWT.core;

import java.awt.image.BufferedImage;

public class WaveletSequentialPipeline implements WaveletPipeline {
    private final HaarWavelet2D haar = new HaarWavelet2D();

    @Override
    public BufferedImage compressAndReconstruct(BufferedImage img, int thresholdPercent) {
        ImageRGB rgb = ImageRGB.fromImage(img);

        float[][] rCoef = haar.forward(rgb.r, -1);
        float[][] gCoef = haar.forward(rgb.g, -1);
        float[][] bCoef = haar.forward(rgb.b, -1);

        float maxAbs = Math.max(MatrixUtil.maxAbs(rCoef),
                Math.max(MatrixUtil.maxAbs(gCoef), MatrixUtil.maxAbs(bCoef)));

        float thr = (thresholdPercent / 100.0f) * (thresholdPercent / 100.0f) * maxAbs;
        MatrixUtil.thresholdInPlace(rCoef, thr);
        MatrixUtil.thresholdInPlace(gCoef, thr);
        MatrixUtil.thresholdInPlace(bCoef, thr);

        float[][] rRec = haar.inverse(rCoef, -1);
        float[][] gRec = haar.inverse(gCoef, -1);
        float[][] bRec = haar.inverse(bCoef, -1);

        return ImageRGB.toImage(rRec, gRec, bRec);
    }
}