package de.schmiereck;

import de.schmiereck.smalllamo.SmallLanguageModel;
import java.util.Random;

//TIP To <b>Run</b> code, press <shortcut actionId="Run"/> or
// click the <icon src="AllIcons.Actions.Execute"/> icon in the gutter.
public class Main {
    public static void main(String[] args) {
        System.out.println("Starte SmallLaMo - Small Language Model für Efficient Inference");

        // Trainingsdaten erstellen (einfache Sequenz: "ae aa ea aa")
        StringBuilder trainingData = new StringBuilder();
        String[] patterns = {"ae ", "ea ", "aa ", "ee "};
        Random random = new Random(42);

        // Zufällige Reihenfolge der Muster generieren
        for (int i = 0; i < 40; i++) {
            trainingData.append(patterns[random.nextInt(patterns.length)]);
        }

        String trainText = trainingData.toString();
        System.out.println("Trainingsdaten: " + trainText);

        // Sprachmodell initialisieren und trainieren
        SmallLanguageModel model = new SmallLanguageModel();
        int epochs = 100;
        System.out.println("Starte Training für " + epochs + " Epochen...");
        model.train(trainText, epochs);

        // Vorhersagen testen
        System.out.println("\nVorhersage-Tests:");
        char[] testChars = {'a', 'e', ' ', 'a', 'a', ' ', 'e', 'e', ' '};
        for (char testChar : testChars) {
            char prediction = model.predict(testChar);
            System.out.println("Zeichen '" + testChar + "' -> Vorhersage: '" + prediction + "'");
        }

        // Text generieren
        System.out.println("\nGenerierter Text:");
        String generatedText = model.generateText('a', 20);
        System.out.println(generatedText);

        // Erweitertes Training mit mehr Zeichen, wenn gewünscht
        /*
        System.out.println("\nErweitertes Training mit 'aeiou':");
        String extendedText = "aeiou aeiou aeiou";
        model.train(extendedText, 50);

        System.out.println("\nGenerierter Text nach erweitertem Training:");
        generatedText = model.generateText('a', 20);
        System.out.println(generatedText);
        */
    }
}