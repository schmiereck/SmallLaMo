package de.schmiereck.smalllamo;

import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.BackpropType;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.LSTM;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Implementierung eines kleinen Sprachmodells mit rekurrenten neuronalen Netzwerken.
 * Das Modell verwendet ein LSTM-Netzwerk mit internem Kontext-Gedächtnis, umgeben von
 * regulären Feed-Forward-Schichten, um bei gegebener Eingabe das nächste Zeichen vorherzusagen.
 */
public class SmallLanguageModel {

    private static final Logger logger = LoggerFactory.getLogger(SmallLanguageModel.class);

    private final int inputSize = 8;        // 8 Bits für jeden Zeichen-Input
    private final int outputSize = 8;       // 8 Bits für die Vorhersage des nächsten Zeichens
    private final int hiddenLayerSize = 32; // Größe der LSTM-Schicht
    private final int preLayerSize1 = 16;   // Größe der ersten versteckten Schicht vor LSTM
    private final int preLayerSize2 = 24;   // Größe der zweiten versteckten Schicht vor LSTM
    private final int postLayerSize = 24;   // Größe der versteckten Schicht nach LSTM
    private final ComputationGraph network;

    /**
     * Konstruktor für das Sprachmodell.
     * Initialisiert die Netzwerkkonfiguration mit mehreren versteckten Schichten
     * und einer LSTM-Schicht für Kontextverarbeitung.
     */
    public SmallLanguageModel() {
        // Konfiguration des neuronalen Netzwerks als ComputationGraph für flexiblere Architektur
        ComputationGraphConfiguration config = new NeuralNetConfiguration.Builder()
                .seed(123)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(new Adam(0.005))
                .weightInit(WeightInit.XAVIER)
                .graphBuilder()

                // Eingabeschicht definieren
                .addInputs("input")

                // Erste versteckte Feed-Forward Schicht
                .addLayer("dense1", new DenseLayer.Builder()
                        .nIn(inputSize)
                        .nOut(preLayerSize1)
                        .activation(Activation.RELU)
                        .build(), "input")

                // Zweite versteckte Feed-Forward Schicht
                .addLayer("dense2", new DenseLayer.Builder()
                        .nIn(preLayerSize1)
                        .nOut(preLayerSize2)
                        .activation(Activation.RELU)
                        .build(), "dense1")

                // LSTM-Schicht für Kontext-Gedächtnis
                .addLayer("lstm", new LSTM.Builder()
                        .nIn(preLayerSize2)
                        .nOut(hiddenLayerSize)
                        .activation(Activation.TANH)
                        .build(), "dense2")

                // Versteckte Feed-Forward Schicht nach LSTM
                .addLayer("dense3", new DenseLayer.Builder()
                        .nIn(hiddenLayerSize)
                        .nOut(postLayerSize)
                        .activation(Activation.RELU)
                        .build(), "lstm")

                // Ausgabeschicht
                .addLayer("output", new RnnOutputLayer.Builder()
                        .nIn(postLayerSize)
                        .nOut(outputSize)
                        .activation(Activation.SIGMOID)
                        .lossFunction(LossFunctions.LossFunction.MSE)
                        .build(), "dense3")

                // Ausgabeschicht festlegen
                .setOutputs("output")

                // Eingabetyp festlegen - Zeitreihe mit 8 Features
                .setInputTypes(InputType.recurrent(inputSize))

                // Erlaubt die Rückwärtspropagation durch die Zeit für Kontext-Gedächtnis
                .backpropType(BackpropType.TruncatedBPTT)
                .tBPTTForwardLength(10)
                .tBPTTBackwardLength(10)

                .build();

        // Netzwerk initialisieren
        network = new ComputationGraph(config);
        network.init();
        network.setListeners(new ScoreIterationListener(20)); // Log-Ausgaben alle 20 Iterationen

        logger.info("Gemischtes neuronales Netzwerk initialisiert mit Architektur: {} -> {} -> {} -> {} -> {} -> {}",
                inputSize, preLayerSize1, preLayerSize2, hiddenLayerSize, postLayerSize, outputSize);
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
        int timeSeriesLength = Math.min(10, characters.length - 1);

        for (int epoch = 0; epoch < epochs; epoch++) {
            double totalLoss = 0.0;

            // Für jede mögliche Startposition der Sequenz
            for (int seqStart = 0; seqStart <= characters.length - timeSeriesLength - 1; seqStart++) {
                network.rnnClearPreviousState();

                // Ein 3D-Array für die Eingabe erstellen [miniBatchSize=1, nIn=inputSize, timeSeriesLength]
                INDArray input = Nd4j.zeros(1, inputSize, timeSeriesLength);
                INDArray labels = Nd4j.zeros(1, outputSize, timeSeriesLength);

                // Die Zeitreihe füllen
                for (int timeStep = 0; timeStep < timeSeriesLength; timeStep++) {
                    char currentChar = characters[seqStart + timeStep];
                    char nextChar = characters[seqStart + timeStep + 1];

                    double[] inputVector = CharEncoder.encode(currentChar);
                    double[] targetVector = CharEncoder.encode(nextChar);

                    // Eingabevektoren für diesen Zeitschritt einfügen
                    for (int i = 0; i < inputSize; i++) {
                        input.putScalar(new int[]{0, i, timeStep}, inputVector[i]);
                    }

                    // Zielvektoren für diesen Zeitschritt einfügen
                    for (int i = 0; i < outputSize; i++) {
                        labels.putScalar(new int[]{0, i, timeStep}, targetVector[i]);
                    }
                }

                // Training für die gesamte Sequenz
                network.fit(new DataSet(input, labels));

                // Vorhersage für die Sequenz
                INDArray output = network.output(input)[0];

                // Verlust berechnen
                double sequenceLoss = 0;
                for (int timeStep = 0; timeStep < timeSeriesLength; timeStep++) {
                    INDArray actualOutput = output.get(NDArrayIndex.point(0), NDArrayIndex.all(), NDArrayIndex.point(timeStep));
                    INDArray expectedOutput = labels.get(NDArrayIndex.point(0), NDArrayIndex.all(), NDArrayIndex.point(timeStep));
                    sequenceLoss += calculateLoss(actualOutput, expectedOutput);
                }

                totalLoss += sequenceLoss / timeSeriesLength;
            }

            // Durchschnittlicher Verlust über alle Sequenzen
            int numSequences = characters.length - timeSeriesLength;
            if (numSequences > 0) {
                logger.info("Epoche {} abgeschlossen, Durchschnittsverlust: {}", epoch, totalLoss / numSequences);
            }
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
        INDArray outputArray = network.output(inputArray)[0];

        // Umwandeln der Ausgabe in ein Zeichen
        double[] outputVector = outputArray.reshape(outputSize).toDoubleVector();
        double[] binaryOutput = CharEncoder.argmax(outputVector);

        return CharEncoder.decode(binaryOutput);
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
     * Gibt das zugrunde liegende neuronale Netzwerk zurück.
     */
    public ComputationGraph getNetwork() {
        return network;
    }
}
