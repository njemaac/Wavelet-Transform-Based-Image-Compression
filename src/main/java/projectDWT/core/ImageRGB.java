package projectDWT.core;

import java.awt.*;
import java.awt.image.BufferedImage;

public class ImageRGB {
    public final float[][] r;
    public final float[][] g;
    public final float[][] b;

    public ImageRGB(float[][] r, float[][] g, float[][] b) {
        this.r = r;
        this.g = g;
        this.b = b;
    }

    public static ImageRGB fromImage(BufferedImage img) {
        int w = img.getWidth();
        int h = img.getHeight();

        float[][] R = new float[h][w];
        float[][] G = new float[h][w];
        float[][] B = new float[h][w];

        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                int rgb = img.getRGB(x, y);
                R[y][x] = (rgb >> 16) & 0xFF;
                G[y][x] = (rgb >> 8) & 0xFF;
                B[y][x] = rgb & 0xFF;
            }
        }
        return new ImageRGB(R, G, B);
    }

    public static BufferedImage toImage(float[][] R, float[][] G, float[][] B) {
        int h = R.length;
        int w = R[0].length;
        BufferedImage out = new BufferedImage(w, h, BufferedImage.TYPE_INT_RGB);

        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                int rr = MatrixUtil.clampToByte(R[y][x]);
                int gg = MatrixUtil.clampToByte(G[y][x]);
                int bb = MatrixUtil.clampToByte(B[y][x]);
                out.setRGB(x, y, (rr << 16) | (gg << 8) | bb);
            }
        }
        return out;
    }
}