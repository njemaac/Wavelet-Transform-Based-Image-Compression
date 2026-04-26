package projectDWT.app;

import projectDWT.core.WaveletPipeline;
import projectDWT.core.WaveletSequentialPipeline;
import projectDWT.core.ParallelWaveletPipeline;
import util.Logger;

import javax.imageio.ImageIO;
import javax.swing.*;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.IOException;

public class WaveletApp {

    private final JFrame frame;
    private final JLabel originalLabel = new JLabel();
    private final JLabel compressedLabel = new JLabel();
    private final JSlider thresholdSlider = new JSlider(0, 100, 5);
    private final JLabel thresholdValueLabel = new JLabel("Threshold: 5%");
    private final JLabel timeLabel = new JLabel("Time: - ms");
    private final JButton loadButton = new JButton("Load Image");
    private final JButton saveButton = new JButton("Save Reconstructed");
    private final JComboBox<String> modeCombo = new JComboBox<>(new String[]{"Sequential", "Parallel"});

    private BufferedImage originalImage;
    private BufferedImage reconstructedImage;
    private WaveletPipeline currentPipeline;

    public WaveletApp() {
        frame = new JFrame("Wavelet DWT Image Compression");
        initUI();
    }

    private void initUI() {
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        frame.setLayout(new BorderLayout());
        JPanel control = new JPanel(new FlowLayout(FlowLayout.LEFT));

        control.add(new JLabel("Mode:"));
        control.add(modeCombo);

        control.add(loadButton);

        control.add(new JLabel("Threshold:"));
        control.add(thresholdSlider);
        control.add(thresholdValueLabel);

        control.add(saveButton);
        control.add(timeLabel);

        frame.add(control, BorderLayout.NORTH);

        JPanel imagesPanel = new JPanel(new GridLayout(1, 2));
        originalLabel.setBorder(BorderFactory.createTitledBorder("Original"));
        compressedLabel.setBorder(BorderFactory.createTitledBorder("Reconstructed"));
        imagesPanel.add(new JScrollPane(originalLabel));
        imagesPanel.add(new JScrollPane(compressedLabel));
        frame.add(imagesPanel, BorderLayout.CENTER);

        loadButton.addActionListener(e -> onLoadImage());
        saveButton.addActionListener(e -> onSaveImage());

        thresholdSlider.addChangeListener(e -> {
            thresholdValueLabel.setText("Threshold: " + thresholdSlider.getValue() + "%");
            if (!thresholdSlider.getValueIsAdjusting() && originalImage != null) {
                runCompression();
            }
        });

        modeCombo.addActionListener(e -> {
            if (originalImage != null) runCompression();
        });

        frame.setSize(1250, 780);
        frame.setLocationRelativeTo(null);
        frame.setVisible(true);

        Logger.info("WaveletApp started (Sequential + Parallel only). Processors: "
                + Runtime.getRuntime().availableProcessors());
    }

    private void onLoadImage() {
        JFileChooser chooser = new JFileChooser();
        if (chooser.showOpenDialog(frame) != JFileChooser.APPROVE_OPTION) return;

        try {
            originalImage = ImageIO.read(chooser.getSelectedFile());
            originalLabel.setIcon(new ImageIcon(scaleImage(originalImage)));
            Logger.info("Loaded image: " + chooser.getSelectedFile().getName());
            runCompression();
            saveButton.setEnabled(true);
        } catch (Exception ex) {
            Logger.error("Failed to load image: " + ex.getMessage());
            JOptionPane.showMessageDialog(frame, "Cannot load image.");
        }
    }

    private void onSaveImage() {
        if (reconstructedImage == null) return;
        JFileChooser chooser = new JFileChooser();
        if (chooser.showSaveDialog(frame) != JFileChooser.APPROVE_OPTION) return;

        try {
            ImageIO.write(reconstructedImage, "png", chooser.getSelectedFile());
            Logger.info("Image saved successfully.");
        } catch (IOException ex) {
            Logger.error("Save failed: " + ex.getMessage());
        }
    }

    private void runCompression() {
        if (originalImage == null) return;

        String mode = (String) modeCombo.getSelectedItem();
        Logger.info("Running " + mode + " mode | Threshold = " + thresholdSlider.getValue() + "%");

        try {
            long start = System.currentTimeMillis();

            currentPipeline = mode.equals("Sequential")
                    ? new WaveletSequentialPipeline()
                    : new ParallelWaveletPipeline();

            reconstructedImage = currentPipeline.compressAndReconstruct(originalImage, thresholdSlider.getValue());

            long elapsed = System.currentTimeMillis() - start;

            timeLabel.setText("Time: " + elapsed + " ms");

            originalLabel.setBorder(BorderFactory.createTitledBorder("Original"));
            compressedLabel.setBorder(BorderFactory.createTitledBorder(
                    mode + " | Threshold " + thresholdSlider.getValue() + "%"));

            originalLabel.setIcon(new ImageIcon(scaleImage(originalImage)));
            compressedLabel.setIcon(new ImageIcon(scaleImage(reconstructedImage)));

        } catch (Exception ex) {
            Logger.error("Compression error: " + ex.getMessage());
            JOptionPane.showMessageDialog(frame, ex.getMessage());
        }
    }

    private Image scaleImage(BufferedImage img) {
        int target = 560;
        float aspect = (float) img.getWidth() / img.getHeight();
        int w = target;
        int h = (int) (w / aspect);
        if (h > target) {
            h = target;
            w = (int) (h * aspect);
        }
        return img.getScaledInstance(w, h, Image.SCALE_SMOOTH);
    }

    public static void main(String[] args) {
        SwingUtilities.invokeLater(WaveletApp::new);
    }
}