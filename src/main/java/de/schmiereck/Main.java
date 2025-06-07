package de.schmiereck;

import de.schmiereck.smalllamo.SmallLanguageModel;
import java.util.Random;

//TIP To <b>Run</b> code, press <shortcut actionId="Run"/> or
// click the <icon src="AllIcons.Actions.Execute"/> icon in the gutter.
public class Main {
    public static void main(String[] args) {
        // Konfigurieren von ND4J für die Nutzung mehrerer CPU-Kerne
        // Setzen Sie die Anzahl der Threads auf die Anzahl der verfügbaren Prozessoren oder einen gewünschten Wert.
        // System.setProperty("org.nd4j.linalg.cpu.omp.maxthreads", String.valueOf(Runtime.getRuntime().availableProcessors()));
        // Alternativ eine feste Anzahl, z.B. 4 Threads:
        System.setProperty("org.nd4j.linalg.cpu.omp.maxthreads", "8");

        System.out.println("Starte SmallLaMo - Small Language Model für Efficient Inference");

        // Trainingsdaten erstellen (einfache Sequenz)
        //String[] patterns = { "0123." };
        //String[] patterns = { "01.", "12.", "23." };
        //String[] patterns = { "ae ", "ea ", "aa ", "ee " };
        //String[] patterns = { "1:ae. ", "2:ea. " };
        //String[] patterns = { "1: ae ea. ", "2: ea ae. ", "3: iu ui. ", "4: ui iu. " };
        Random random = new Random(42);

        SmallLanguageModel model = new SmallLanguageModel();

        {
            String[] patterns = { "01." };
            int epochs = 10;
            trainSmallLanguageModel(model, random, patterns, epochs);
            generateText(model);
        }
        {
            String[] patterns = { "11." };
            int epochs = 100;
            trainSmallLanguageModel(model, random, patterns, epochs);
            generateText(model);
        }
//        {
//            String[] patterns = { "00." };
//            //String[] patterns = { "01" };
//            int epochs = 200;
//            trainSmallLanguageModel(model, random, patterns, epochs);
//            generateText(model);
//        }
//        {
//            String[] patterns = { "00.11." };
//            int epochs = 100;
//            trainSmallLanguageModel(model, random, patterns, epochs);
//            generateText(model);
//        }
//        {
//            String[] patterns = { "00.11.22." };
//            int epochs = 100;
//            trainSmallLanguageModel(model, random, patterns, epochs);
//            generateText(model);
//        }
//        {
//            String[] patterns = { "00.11.22." };
//            int epochs = 100;
//            trainSmallLanguageModel(model, random, patterns, epochs);
//            generateText(model);
//        }
//        {
//            String[] patterns = { "00.", "11.", "22." };
//            int epochs = 100;
//            trainSmallLanguageModel(model, random, patterns, epochs);
//            generateText(model);
//        }
    }

    private static void generateText(SmallLanguageModel model) {
        {
            System.out.println("\nGenerierter Text:");
            String generatedText = model.generateText('0', 100);
            System.out.println(generatedText);
        }
        {
            System.out.println("\nGenerierter Text:");
            String generatedText = model.generateText('1', 100);
            System.out.println(generatedText);
        }
        {
            System.out.println("\nGenerierter Text:");
            String generatedText = model.generateText('2', 100);
             System.out.println(generatedText);
        }
    }

    private static SmallLanguageModel trainSmallLanguageModel(SmallLanguageModel model, Random random, String[] patterns, int epochs) {
        // Zufällige Reihenfolge der Muster generieren
        StringBuilder trainingData = new StringBuilder();
        for (int i = 0; i < 50; i++) {
            trainingData.append(patterns[random.nextInt(patterns.length)]);
        }

        String trainText = trainingData.toString();
        System.out.println("Trainingsdaten: " + trainText);

        // Sprachmodell initialisieren und trainieren
        System.out.println("Starte Training für " + epochs + " Epochen...");
        model.train(trainText, epochs);
        return model;
    }
}