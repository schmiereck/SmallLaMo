package de.schmiereck.smalllamo;

import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.BackpropType;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.LSTM;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.conf.preprocessor.RnnToFeedForwardPreProcessor; // Import für manuellen Preprocessor
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Implementierung eines kleinen Sprachmodells mit rekurrenten neuronalen Netzwerken.
 * Das Modell verwendet ein LSTM-Netzwerk mit internem Kontext-Gedächtnis, umgeben von
 * Dense-Layern, um bei gegebener Eingabe das nächste Zeichen vorherzusagen.
 */
public class SmallLanguageModel {

    private static final Logger logger = LoggerFactory.getLogger(SmallLanguageModel.class);

    private final int inputSize = 8;        // 8 Bits für jeden Zeichen-Input
    private final int preLayerSize1 = 8 * 3;   // Größe der ersten versteckten Schicht vor LSTM
    private final int preLayerSize2 = 8 * 6;   // Größe der zweiten versteckten Schicht vor LSTM
    private final int lstmLayerSize = 8 * 10; // Größe der LSTM-Schicht
    private final int postLayer1Size = 8 * 6;   // Größe der versteckten Schicht nach LSTM
    private final int postLayer2Size = 8 * 6;   // Größe der versteckten Schicht nach LSTM
    private final int outputSize = 8;       // 8 Bits für die Vorhersage des nächsten Zeichens
    private final MultiLayerNetwork network;

    /**
     * Konstruktor für das Sprachmodell.
     * Initialisiert die Netzwerkkonfiguration mit mehreren versteckten Schichten
     * und einer LSTM-Schicht für Kontextverarbeitung.
     */
    public SmallLanguageModel() {
        // Konfiguration des neuronalen Netzwerks mit aufeinanderfolgenden Schichten
        MultiLayerConfiguration config = new NeuralNetConfiguration.Builder()
                .seed(123)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(new Adam(0.005))
                .weightInit(WeightInit.XAVIER)
                .list()
                // Erste versteckte Feed-Forward Schicht vor LSTM
                // WICHTIG: Bei RNN-Architekturen werden die Feedforward-Layer für jeden Zeitschritt angewendet
                .layer(0, new DenseLayer.Builder()
                        .nIn(inputSize)
                        .nOut(preLayerSize1)
                        .activation(Activation.RELU)
                        .build())
                // Zweite versteckte Feed-Forward Schicht vor LSTM
                .layer(1, new DenseLayer.Builder()
                        .nIn(preLayerSize1)
                        .nOut(preLayerSize2)
                        .activation(Activation.RELU)
                        .build())
                // LSTM-Schicht für Kontext-Gedächtnis
                .layer(2, new LSTM.Builder()
                        .nIn(preLayerSize2)
                        .nOut(lstmLayerSize)
                        .activation(Activation.TANH)
                        .build())
                // Versteckte Feed-Forward Schicht nach LSTM
                .layer(3, new DenseLayer.Builder()
                        .nIn(lstmLayerSize)
                        .nOut(postLayer1Size)
                        .activation(Activation.RELU)
                        .build())
                .layer(4, new DenseLayer.Builder()
                        .nIn(postLayer1Size)
                        .nOut(postLayer2Size)
                        .activation(Activation.RELU)
                        .build())
                // Ausgabeschicht
                .layer(5, new RnnOutputLayer.Builder()
                        .nIn(postLayer2Size)
                        .nOut(outputSize)
                        .activation(Activation.SIGMOID)
                        .lossFunction(LossFunctions.LossFunction.MSE)
                        .build())
                // Erlaubt die Rückwärtspropagation durch die Zeit für Kontext-Gedächtnis
                .backpropType(BackpropType.TruncatedBPTT)
                .tBPTTForwardLength(10)
                .tBPTTBackwardLength(10)
                .setInputType(InputType.recurrent(inputSize)) // Hinzugefügt für korrekte Eingabebehandlung
                .inputPreProcessor(0, new RnnToFeedForwardPreProcessor()) // Korrigierte Position für den Preprocessor
                .build();

        // Netzwerk initialisieren
        network = new MultiLayerNetwork(config);
        network.init();
        network.setListeners(new ScoreIterationListener(1000)); // Log-Ausgaben alle 20 Iterationen

        logger.info("Erweitertes neuronales Netzwerk initialisiert mit Architektur: {} -> {} -> {} -> {} -> {} -> {}",
                inputSize, preLayerSize1, preLayerSize2, lstmLayerSize, postLayer1Size, postLayer2Size, outputSize);
    }

    /**
     * Trainiert das Modell mit einer Textsequenz.
     *
     * @param text Der Trainingstext
     * @param epochs Anzahl der Trainingsepochen
     */
    public void train(String text, int epochs) {
        logger.info("Starte Training mit Text: '{}' für {} Epochen", text, epochs);

        char[] characters = text.toCharArray();

        // Länge der Sequenz für BPTT festlegen - z.B. 10 Zeichen
        int timeSeriesLength = characters.length - 1; //Math.min(9, characters.length - 1);

        for (int epoch = 0; epoch < epochs; epoch++) {
            network.rnnClearPreviousState();
            double sequenceLoss = 0;

            for (int charPos = 0; charPos < timeSeriesLength; charPos++) {
                char currentChar = characters[charPos];
                char nextChar = characters[charPos + 1];

                // Ein 3D-Array für die Eingabe erstellen [miniBatchSize=1, nIn=inputSize, timeSeriesLength]
                INDArray inputArr = Nd4j.zeros(1, inputSize, 1);
                INDArray outputArr = Nd4j.zeros(1, outputSize, 1);

                double[] inputVector = CharEncoder.encode(currentChar);
                double[] targetVector = CharEncoder.encode(nextChar);

                // Eingabevektoren für diesen Zeitschritt einfügen
                for (int inputPos = 0; inputPos < inputSize; inputPos++) {
                    inputArr.putScalar(new int[]{0, inputPos, 0}, inputVector[inputPos]);
                }

                // Zielvektoren für diesen Zeitschritt einfügen
                for (int outputPos = 0; outputPos < outputSize; outputPos++) {
                    outputArr.putScalar(new int[]{0, outputPos, 0}, targetVector[outputPos]);
                }

                // Training für die gesamte Sequenz
                network.fit(inputArr, outputArr);

                // Vorhersage für die Sequenz
                INDArray output = network.output(inputArr);

                // Verlust berechnen
                //for (int charPos = 0; charPos < timeSeriesLength; charPos++) {
                    INDArray actualOutput = output.get(NDArrayIndex.point(0), NDArrayIndex.all(), NDArrayIndex.point(0));
                    INDArray expectedOutput = outputArr.get(NDArrayIndex.point(0), NDArrayIndex.all(), NDArrayIndex.point(0));
                    sequenceLoss += calculateLoss(actualOutput, expectedOutput);
                //}
            }

            // Durchschnittlicher Verlust über alle Sequenzen
            logger.info("Epoche {} abgeschlossen, Durchschnittsverlust: {}", epoch, sequenceLoss / timeSeriesLength);
        }

        logger.info("Training abgeschlossen.");
    }

    /**
     * Berechnet den Verlust zwischen Vorhersage und Zielwert.
     */
    private double calculateLoss(INDArray output, INDArray target) {
        INDArray diff = target.sub(output);
        return diff.mul(diff).sumNumber().doubleValue();
    }

    /**
     * Generiert eine Textsequenz beginnend mit einem Startzeichen.
     *
     * @param startChar Das Startzeichen
     * @param length Die Länge der zu generierenden Sequenz
     * @return Die generierte Textsequenz
     */
    public String generateText(char startChar, int length) {
        StringBuilder result = new StringBuilder();
        result.append(startChar);

        char currentChar = startChar;

        // Reset des Netzwerkzustands vor der Generierung
        network.rnnClearPreviousState();

        for (int i = 0; i < length - 1; i++) {
            currentChar = predict(currentChar);
            result.append(currentChar);
        }

        return result.toString();
    }

    /**
     * Vorhersage des nächsten Zeichens bei gegebener Eingabe.
     *
     * @param input Das Eingabezeichen
     * @return Das vorhergesagte nächste Zeichen
     */
    public char predict(char input) {
        // Eingabezeichen kodieren
        double[] inputVector = CharEncoder.encode(input);
        INDArray inputArray = Nd4j.create(inputVector).reshape(1, inputSize, 1);

        // Vorhersage durch das Netzwerk
        INDArray outputArray = network.output(inputArray);

        // Umwandeln der Ausgabe in ein Zeichen
        // Da outputArray 3D ist [1, outputSize, 1], müssen wir den Vektor für den ersten Batch und ersten Zeitschritt extrahieren.
        INDArray outputVectorSlice = outputArray.get(NDArrayIndex.point(0), NDArrayIndex.all(), NDArrayIndex.point(0));
        double[] outputVector = outputVectorSlice.toDoubleVector();
        double[] binaryOutput = CharEncoder.argmax(outputVector);

        return CharEncoder.decode(binaryOutput);
    }

    /**
     * Gibt das zugrunde liegende neuronale Netzwerk zurück.
     */
    public MultiLayerNetwork getNetwork() {
        return network;
    }
}
