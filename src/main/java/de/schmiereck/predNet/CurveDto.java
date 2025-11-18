package de.schmiereck.predNet;

public class CurveDto {
    private final int[] inputArr;

    public CurveDto(final int[] inputArr) {
        this.inputArr = inputArr;
    }

    public int[] getInputArr() {
        return this.inputArr;
    }
}
