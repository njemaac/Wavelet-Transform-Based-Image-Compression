package projectDWT.core;
import java.awt.*;
import java.awt.image.BufferedImage;

public class ImageRGB {
    public final double[][] r;
    public final double[][] g;
    public final double[][] b;

    public ImageRGB(double[][] r, double[][] g, double[][] b) {
        this.r = r;
        this.g = g;
        this.b = b;
    }

    public static ImageRGB fromImage(BufferedImage img){
        int w = img.getWidth();
        int h = img.getHeight();

        double[][] R = new double[h][w];
        double[][] G = new double[h][w];
        double[][] B = new double[h][w];

        for (int y = 0; y < h; y++){
            for (int x = 0; x < w; x++){
                int rgb = img.getRGB(x, y);
                Color c = new Color(rgb);
                R[y][x] = c.getRed();
                G[y][x] = c.getGreen();
                B[y][x] = c.getBlue();
            }
        }
        return new ImageRGB(R, G, B);
    }

    public static BufferedImage toImage(double[][] R, double[][] G, double[][] B){
        int h = R.length;
        int w = R[0].length;
        BufferedImage out = new BufferedImage(w, h, BufferedImage.TYPE_INT_RGB);

        for (int y = 0; y < h; y++){
            for (int x = 0; x<w; x++){
                int rr = MatrixUtil.clampToByte(R[y][x]);
                int gg = MatrixUtil.clampToByte(G[y][x]);
                int bb = MatrixUtil.clampToByte(B[y][x]);
                out.setRGB(x,y, new Color(rr, gg, bb).getRGB());
            }
        }
        return out;
    }
}
