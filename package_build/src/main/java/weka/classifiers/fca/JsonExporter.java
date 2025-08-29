package weka.classifiers.fca;
import weka.classifiers.fca._GaloisConnection.FormalConcept;
import weka.classifiers.rules.CNC;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.TreeMap;

public class JsonExporter {

    public static String toJson(List<_GaloisConnection.FormalConcept> conceptsList) {
        if (conceptsList == null || conceptsList.isEmpty()) {
            return "[]";
        }

        StringBuilder jsonBuilder = new StringBuilder("[");
        for (FormalConcept concept : conceptsList) {
            jsonBuilder.append(String.format(
                "{\"extent\": %s, \"intent\": %s},",
                setToJsonInt(concept.getObjects()),
                setToJsonStrings(concept.getAttributes()) 
            ));
        }
        
        // Supprimer la dernière virgule
        jsonBuilder.deleteCharAt(jsonBuilder.length() - 1);
        jsonBuilder.append("]");
        
        return jsonBuilder.toString();
    }

 // Pour les Set<String> (nouvelle méthode)
    private static String setToJsonStrings(Set<String> set) {
        if (set == null || set.isEmpty()) {
            return "[]";
        }
        
        StringBuilder sb = new StringBuilder("[");
        for (String item : set) {
            sb.append("\"").append(item).append("\",");
        }
        sb.deleteCharAt(sb.length() - 1); // Supprimer la dernière virgule
        sb.append("]");
        return sb.toString();
    }

    // Pour les Set<Integer>
    private static String setToJsonInt(Set<Integer> set) {
        if (set == null || set.isEmpty()) {
            return "[]";
        }
        
        StringBuilder sb = new StringBuilder("[");
        for (Integer item : set) {
            sb.append(item).append(",");
        }
        sb.deleteCharAt(sb.length() - 1); // Supprimer la dernière virgule
        sb.append("]");
        return sb.toString();
    }
    
    public static String toJsonGroupedByLevel(List<FormalConcept> conceptsList) {
        if (conceptsList == null || conceptsList.isEmpty()) {
            return "[]";
        }

        // Grouper les concepts par niveau (nombre d'attributs dans l'intention)
        Map<Integer, List<FormalConcept>> conceptsByLevel = new TreeMap<>();
        for (FormalConcept concept : conceptsList) {
            int level = concept.getAttributes().size();
            conceptsByLevel.computeIfAbsent(level, k -> new ArrayList<>()).add(concept);
        }

        StringBuilder jsonBuilder = new StringBuilder("[");
        int conceptCounter = 1;

        for (Map.Entry<Integer, List<FormalConcept>> entry : conceptsByLevel.entrySet()) {
            int level = entry.getKey();
            List<FormalConcept> conceptsAtLevel = entry.getValue();

            jsonBuilder.append("{");
            jsonBuilder.append("\"level\": ").append(level).append(",");
            jsonBuilder.append("\"concepts\": [");

            for (FormalConcept concept : conceptsAtLevel) {
                String conceptName = "C" + conceptCounter++;
                jsonBuilder.append("{")
                           .append("\"name\": \"").append(conceptName).append("\", ")
                           .append("\"extent\": ").append(setToJsonInt(concept.getObjects())).append(", ")
                           .append("\"intent\": ").append(setToJsonStrings(concept.getAttributes()))
                           .append("},");
            }

            // Supprimer la dernière virgule
            if (!conceptsAtLevel.isEmpty()) {
                jsonBuilder.deleteCharAt(jsonBuilder.length() - 1);
            }

            jsonBuilder.append("]},");
        }

        // Supprimer la dernière virgule finale
        jsonBuilder.deleteCharAt(jsonBuilder.length() - 1);
        jsonBuilder.append("]");

        return jsonBuilder.toString();
    }

    
    public static void saveToFile(String jsonString, String filename) {
        try {
            Files.write(Paths.get(filename), jsonString.getBytes());
            CNC.logDebugFile("JSON file successfully created at: " + filename);
        } catch (IOException e) {
            System.err.println("Error writing to file: " + e.getMessage());
        }
    }
}