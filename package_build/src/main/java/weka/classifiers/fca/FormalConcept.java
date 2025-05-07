package weka.classifiers.fca;

import java.util.ArrayList;

import weka.core.Instances;

public class FormalConcept {

    private ArrayList<Integer> extent;  // Ensemble d'instances
    private ArrayList<String> intent;   // Ensemble d'attributs
    
    public FormalConcept(ArrayList<Integer> extent, ArrayList<String> intent) {
        this.extent = new ArrayList<>(extent);
        this.intent = new ArrayList<>(intent);
    }
    
    // Getters
    public ArrayList<Integer> getExtent() { return extent; }
    public ArrayList<String> getIntent() { return intent; }
    
    // Affichage formaté
    public String toString(Instances inst) {
        StringBuilder sb = new StringBuilder("Concept Formel:\n");
        sb.append("  Extent (Instances): ").append(formatExtent(inst)).append("\n");
        sb.append("  Intent (Attributs): ").append(formatIntent(inst));
        return sb.toString();
    }
    
    private String formatExtent(Instances inst) {
        if (extent.isEmpty()) return "{}";
        
        StringBuilder sb = new StringBuilder("{");
        for (int i = 0; i < extent.size(); i++) {
            if (i > 0) sb.append(", ");
            sb.append("o").append(extent.get(i)+1);  // o1, o2, etc.
        }
        sb.append("}");
        return sb.toString();
    }
    
    private String formatIntent(Instances inst) {
        if (intent.isEmpty()) return "{}";
        
        StringBuilder sb = new StringBuilder("{");
        boolean first = true;
        for (int i = 0; i < intent.size(); i++) {
            if (!intent.get(i).equals("-null-")) {
                if (!first) sb.append(", ");
                sb.append(inst.attribute(i).name())
                  .append("=")
                  .append(intent.get(i));
                first = false;
            }
        }
        sb.append("}");
        return sb.toString();
    }
}