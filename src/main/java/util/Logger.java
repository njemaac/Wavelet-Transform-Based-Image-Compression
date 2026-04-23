package util;

import java.text.SimpleDateFormat;

public class Logger {
    private static final String RED = "\u001B[31m";
    private static final String GREEN =  "\u001B[32m";
    private static final String YELLOW = "\u001B[33m";
    private static final String BLUE = "\u001B[34m";
    private static final String RESET = "\u001B[0m";

    private static final SimpleDateFormat dateFormat = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss");

    public enum LogLevel {
        Debug,
        Info,
        Warn,
        Error
    }
    private static void log(String message, LogLevel level) {
        String date = dateFormat.format(System.currentTimeMillis());
        String name = Thread.currentThread().getName();
        String messagePrefix = "[" + date + "][" + name + "] " + level + " : ";

        switch (level) {
            case Debug -> messagePrefix = BLUE + messagePrefix + RESET;
            case Info -> messagePrefix = GREEN + messagePrefix + RESET;
            case Warn -> messagePrefix = YELLOW + messagePrefix + RESET;
            case Error -> messagePrefix = RED + messagePrefix + RESET;
        }

        System.out.println(messagePrefix + message);
    }

    public static void info(String message) {
        log(message, LogLevel.Info);
    }
    public static void warn(String message) {
        log(message, LogLevel.Warn);
    }
    public static void error(String message) {
        log(message, LogLevel.Error);
    }
    public static void debug(String message) {
        log(message, LogLevel.Debug);
    }
}

