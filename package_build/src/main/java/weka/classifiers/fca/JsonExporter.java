package weka.classifiers.fca;
import weka.classifiers.fca._GaloisConnection.FormalConcept;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.List;
import java.util.Set;

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
    
    public static void saveToFile(String jsonString, String filename) {
        try {
            Files.write(Paths.get(filename), jsonString.getBytes());
            System.out.println("JSON file successfully created at: " + filename);
        } catch (IOException e) {
            System.err.println("Error writing to file: " + e.getMessage());
        }
    }
}