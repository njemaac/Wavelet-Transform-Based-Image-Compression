package projectDWT.app;

import projectDWT.core.ParallelWaveletPipeline;
import util.Logger;

import javax.imageio.ImageIO;
import javax.swing.*;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;

public class WaveletParallelApp {

    private JFrame frame;
    private JLabel originalLabel;
    private JLabel compressedLabel;
    private JSlider thresholdSlider;
    private JLabel timeLabel;
    private JButton loadButton;
    private JButton saveButton;
    private BufferedImage originalImage;
    private BufferedImage reconstructedImage;

    private final ParallelWaveletPipeline pipeline = new ParallelWaveletPipeline();

    public WaveletParallelApp() {
        initUI();
    }

    private void initUI() {
        frame = new JFrame("Wavelet DWT Compression - Parallel");
        frame.setDefaultCloseOperation(3);
        frame.setLayout(new BorderLayout());

        JPanel control = new JPanel(new FlowLayout(FlowLayout.LEFT));

        loadButton = new JButton("Load Image");
        saveButton = new JButton("Save reconstructed");
        saveButton.setEnabled(false);

        thresholdSlider = new JSlider(0, 100, 5);
        thresholdSlider.setMajorTickSpacing(25);
        thresholdSlider.setMinorTickSpacing(5);
        thresholdSlider.setPaintTicks(true);
        thresholdSlider.setPaintLabels(true);

        timeLabel = new JLabel("Time: - ms");

        control.add(loadButton);
        control.add(new JLabel("Threshold %: "));
        control.add(thresholdSlider);
        control.add(saveButton);
        control.add(timeLabel);

        frame.add(control, BorderLayout.NORTH);

        JPanel imagesPanel = new JPanel(new GridLayout(1, 2));

        originalLabel = new JLabel();
        originalLabel.setHorizontalAlignment(JLabel.CENTER);
        originalLabel.setVerticalAlignment(JLabel.CENTER);
        originalLabel.setBorder(BorderFactory.createTitledBorder("Original"));

        compressedLabel = new JLabel();
        compressedLabel.setHorizontalAlignment(JLabel.CENTER);
        compressedLabel.setVerticalAlignment(JLabel.CENTER);
        compressedLabel.setBorder(BorderFactory.createTitledBorder("Reconstructed (Parallel)"));

        imagesPanel.add(new JScrollPane(originalLabel));
        imagesPanel.add(new JScrollPane(compressedLabel));

        frame.add(imagesPanel, BorderLayout.CENTER);

        loadButton.addActionListener(e -> onLoadImage());
        saveButton.addActionListener(e -> onSaveImage());
        thresholdSlider.addChangeListener(e -> {
            if (!thresholdSlider.getValueIsAdjusting() && originalImage != null) {
                runParallel();
            }
        });

        frame.setSize(1000, 700);
        frame.setLocationRelativeTo(null);
        frame.setVisible(true);

        Logger.info("Parallel App started. Available processors: " + Runtime.getRuntime().availableProcessors());
    }

    private void onLoadImage() {
        JFileChooser chooser = new JFileChooser();
        int res = chooser.showOpenDialog(frame);
        if (res != JFileChooser.APPROVE_OPTION) return;

        File f = chooser.getSelectedFile();
        try {
            originalImage = ImageIO.read(f);
            if (originalImage == null) {
                Logger.error("Could not read image.");
                JOptionPane.showMessageDialog(frame, "Unsupported image format.");
                return;
            }
            originalLabel.setIcon(new ImageIcon(scaleForLabel(originalImage, originalLabel)));
            Logger.info("Loaded image: " + f.getAbsolutePath()
                    + " (" + originalImage.getWidth() + "x" + originalImage.getHeight() + ")");

            runParallel();
            saveButton.setEnabled(true);

        } catch (IOException ex) {
            Logger.error("Error loading image: " + ex.getMessage());
            JOptionPane.showMessageDialog(frame, "Error loading image: " + ex.getMessage());
        }
    }

    private void onSaveImage() {
        if (reconstructedImage == null) {
            JOptionPane.showMessageDialog(frame, "No reconstructed image to save.");
            return;
        }
        JFileChooser chooser = new JFileChooser();
        int res = chooser.showSaveDialog(frame);
        if (res != JFileChooser.APPROVE_OPTION) return;

        File out = chooser.getSelectedFile();
        try {
            ImageIO.write(reconstructedImage, "png", out);
            Logger.info("Saved reconstructed image to: " + out.getAbsolutePath());
        } catch (IOException ex) {
            Logger.error("Error saving image: " + ex.getMessage());
            JOptionPane.showMessageDialog(frame, "Error saving image: " + ex.getMessage());
        }
    }

    private void runParallel() {
        int thresholdPercent = thresholdSlider.getValue();
        Logger.info("Starting parallel compression... threshold = " + thresholdPercent + "%");

        new Thread(() -> {
            try {
                long t0 = System.currentTimeMillis();
                BufferedImage result = pipeline.compressAndReconstruct(originalImage, thresholdPercent);
                long elapsed = System.currentTimeMillis() - t0;

                reconstructedImage = result;

                SwingUtilities.invokeLater(() -> {
                    Logger.info("Parallel compression+decompression time: " + elapsed + " ms");
                    timeLabel.setText("Time: " + elapsed + " ms");
                    originalLabel.setIcon(new ImageIcon(scaleForLabel(originalImage, originalLabel)));
                    compressedLabel.setIcon(new ImageIcon(scaleForLabel(reconstructedImage, compressedLabel)));
                    saveButton.setEnabled(true);
                });

            } catch (InterruptedException ex) {
                Logger.error("Parallel processing interrupted: " + ex.getMessage());
            }
        }, "parallel-worker").start();
    }

    private Image scaleForLabel(BufferedImage img, JLabel label) {
        int w = label.getWidth();
        int h = label.getHeight();
        if (w <= 0 || h <= 0) return img.getScaledInstance(400, 400, Image.SCALE_SMOOTH);

        float aspect = (float) img.getWidth() / img.getHeight();

        int nw = w;
        int nh = (int) (nw / aspect);
        if (nh > h) {
            nh = h;
            nw = (int) (nh * aspect);
        }
        return img.getScaledInstance(nw, nh, Image.SCALE_SMOOTH);
    }

    public static void main(String[] args) {
        SwingUtilities.invokeLater(WaveletParallelApp::new);
    }
}