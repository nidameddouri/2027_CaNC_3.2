package weka.classifiers.fca;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.TreeMap;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.stream.Collectors;
import java.util.HashMap;
import java.util.Map;
import weka.core.Attribute;
import weka.core.Instances;


/**
 * Implementation of Galois Connection for Formal Concept Analysis
 */

public class _GaloisConnection {
    
    private boolean m_Debug;
    
    // Constants for Galois connection strategies
    public static final int GALOIS_PERTINENT_MAJORITY = 1;  // Pertinent attribute & Majority Value
    public static final int GALOIS_PERTINENT_ALL = 2;       // Pertinent Attribute & All Values
    public static final int GALOIS_ALL_MAJORITY = 3;        // All Attributes & Majority Value
    public static final int GALOIS_ALL_ALL = 4;             // All Attributes & All Values
    
    // The formal context
    private Instances m_Instances;
    
    // Maps for the Galois connections
    private Map<Set<Integer>, Set<String>> intentMap;  // Object set -> Attribute set
    private Map<Set<String>, Set<Integer>> extentMap;  // Attribute set -> Object set
    
    /**
     * Constructor
     * 
     * @param debug Debug mode flag
     */
    public _GaloisConnection(boolean debug) {
        this.m_Debug = debug;
        this.intentMap = new HashMap<>();
        this.extentMap = new HashMap<>();
    }
    
    /**
     * Build the Galois connection from the given instances
     * 
     * @param instances The training instances
     */
    public void buildConnection(Instances instances) {
        this.m_Instances = instances;
        
        if (m_Debug) {
            System.out.println("Building Galois connection from " + instances.numInstances() + " instances");
        }
        
        // Initialize maps
        this.intentMap.clear();
        this.extentMap.clear();
        
        // Build the formal context
        buildFormalContext();
    }
    
   
    
    
    /**
     * Get the intent (attributes) for a set of objects
     * 
     * @param objects Set of object indices
     * @return Set of common attributes
     */
    public Set<String> getIntent(Set<Integer> objects) {
        if (objects.isEmpty()) {
            // Empty set of objects has all attributes
            Set<String> allAttributes = new HashSet<>();
            for (int j = 0; j < m_Instances.numAttributes(); j++) {
                if (j == m_Instances.classIndex()) continue;
                
                Attribute attr = m_Instances.attribute(j);
                for (int k = 0; k < attr.numValues(); k++) {
                    allAttributes.add(attr.name() + "=" + attr.value(k));
                }
            }
            return allAttributes;
        }
        
        // Get the common attributes of all objects
        Set<String> intent = null;
        
        for (Integer objIdx : objects) {
            Set<Integer> singleObj = new HashSet<>();
            singleObj.add(objIdx);
            
            Set<String> objAttrs = intentMap.get(singleObj);
            
            if (intent == null) {
                intent = new HashSet<>(objAttrs);
            } else {
                intent.retainAll(objAttrs);
            }
        }
        
        return intent;
    }
    
    
    
    /**
     * Get the extent (objects) for a set of attributes (modified to ignore zero/null values)
     * 
     * @param attributes Set of attributes
     * @return Set of objects having all these attributes
     */
    public Set<Integer> getExtent(Set<String> attributes) {
        if (attributes.isEmpty()) {
            // Empty set of attributes has all objects
            Set<Integer> allObjects = new HashSet<>();
            for (int i = 0; i < m_Instances.numInstances(); i++) {
                allObjects.add(i);
            }
            return allObjects;
        }
        
        Set<Integer> extent = new HashSet<>();
        
        for (int i = 0; i < m_Instances.numInstances(); i++) {
            boolean matchesAll = true;
            
            for (String attr : attributes) {
                String[] parts = attr.split("=");
                String attrName = parts[0];
                String attrValue = parts[1];
                
                // Skip if value is zero/null (for binary datasets)
                if (attrValue.equals("0") || attrValue.equalsIgnoreCase("null")) {
                    continue;
                }
                
                // Find attribute index
                int attrIdx = -1;
                for (int j = 0; j < m_Instances.numAttributes(); j++) {
                    if (m_Instances.attribute(j).name().equals(attrName)) {
                        attrIdx = j;
                        break;
                    }
                }
                
                if (attrIdx == -1 || !m_Instances.instance(i).stringValue(attrIdx).equals(attrValue)) {
                    matchesAll = false;
                    break;
                }
            }
            
            if (matchesAll) {
                extent.add(i);
            }
        }
        
        return extent;
    }

    /**
     * Build the formal context from the instances (modified to ignore zero/null values)
     */
    private void buildFormalContext() {
        for (int i = 0; i < m_Instances.numInstances(); i++) {
            Set<String> attributeSet = new HashSet<>();
            
            for (int j = 0; j < m_Instances.numAttributes(); j++) {
                if (j == m_Instances.classIndex()) continue;
                
                Attribute attr = m_Instances.attribute(j);
                String attrValue = m_Instances.instance(i).stringValue(attr);
                
                // Skip zero/null values for binary attributes
                if (attr.isNominal() && (attrValue.equals("0") || attrValue.equalsIgnoreCase("null"))) {
                    continue;
                }
                
                attributeSet.add(attr.name() + "=" + attrValue);
            }
            
            Set<Integer> objectSet = new HashSet<>();
            objectSet.add(i);
            intentMap.put(objectSet, attributeSet);
            
            if (extentMap.containsKey(attributeSet)) {
                extentMap.get(attributeSet).add(i);
            } else {
                Set<Integer> newSet = new HashSet<>();
                newSet.add(i);
                extentMap.put(attributeSet, newSet);
            }
        }
        
        if (m_Debug) {
            System.out.println("Initial formal context built (ignoring zero/null values)");
            System.out.println("Intent map size: " + intentMap.size());
            System.out.println("Extent map size: " + extentMap.size());
            
        }
    }    
    

    /**
     * Find the majority class for a set of objects
     * 
     * @param objects Set of object indices
     * @return Index of the majority class
     */
    private int findMajorityClass(Set<Integer> objects) {
        int[] classCounts = new int[m_Instances.numClasses()];
        
        for (Integer objIdx : objects) {
            int classIdx = (int) m_Instances.instance(objIdx).classValue();
            classCounts[classIdx]++;
        }
        
        int majorityClassIdx = 0;
        for (int i = 0; i < classCounts.length; i++) {
            if (classCounts[i] > classCounts[majorityClassIdx]) {
                majorityClassIdx = i;
            }
        }
        
        return majorityClassIdx;
    }
    
    /**
     * Convert a set of attribute=value pairs to the format used by Classification_Rule
     * 
     * @param attributeSet Set of "attribute=value" strings
     * @return ArrayList of values in the order of attributes in the dataset
     */
    private ArrayList<String> convertAttributesToRuleFormat(Set<String> attributeSet) {
        ArrayList<String> result = new ArrayList<>();
        
        // Initialize with null values
        for (int i = 0; i < m_Instances.numAttributes() - 1; i++) {
            result.add("-null-");
        }
        
        // Fill in values from the attribute set
        for (String attrVal : attributeSet) {
            String[] parts = attrVal.split("=");
            String attrName = parts[0];
            String value = parts[1];
            
            // Find attribute index
            for (int i = 0; i < m_Instances.numAttributes(); i++) {
                if (i == m_Instances.classIndex()) continue;
                
                if (m_Instances.attribute(i).name().equals(attrName)) {
                    int idx = (i < m_Instances.classIndex()) ? i : i - 1;
                    result.set(idx, value);
                    break;
                }
            }
        }
        
        return result;
    }
    
    /**
     * Calculate the weight (coverage ratio) of a rule
     * 
     * @param rule The rule to evaluate
     * @return The weight value
     */
    private double calculateRuleWeight(Classification_Rule rule) {
        double weight = 0;
        
        try {
            Classify_Instance classifier = new Classify_Instance();
            
            for (int i = 0; i < m_Instances.numInstances(); i++) {
                if (classifier.TestClassifInstance_nom(m_Instances.instance(i), rule)) {
                    weight++;
                }
            }
        } catch (Exception e) {
            if (m_Debug) {
                System.err.println("Error calculating rule weight: " + e.getMessage());
            }
            return 0;
        }
        
        return weight / m_Instances.numInstances();
    }
    
    /**
     * Class representing a formal concept (pair of object set and attribute set)
     */
    public class FormalConcept {
        private Set<Integer> objects;
        private Set<String> attributes;
        
        public FormalConcept(Set<Integer> objects, Set<String> attributes) {
            this.objects = objects;
            this.attributes = attributes;
        }
        
        public Set<Integer> getObjects() {
            return objects;
        }
        
        public Set<String> getAttributes() {
            return attributes;
        }
        
        @Override
        public boolean equals(Object obj) {
            if (!(obj instanceof FormalConcept)) return false;
            
            FormalConcept other = (FormalConcept) obj;
            return this.objects.equals(other.objects) && this.attributes.equals(other.attributes);
        }
        
        @Override
        public int hashCode() {
            return objects.hashCode() * 31 + attributes.hashCode();
        }
        
        @Override
        public String toString() {
            return "Concept: [Objects: " + objects.size() + ", Attributes: " + attributes.size() + "]";
        }
    }
    
    /**
     * Génère les concepts formels à partir des instances et attributs
     */
    ArrayList<Classification_Rule> generateConcepts(Instances inst, 
                                                         ArrayList<Attribute> attributes,
                                                         ArrayList<String> allConcepts) throws Exception {
        Map<Set<Integer>, Classification_Rule> bestConcepts = new HashMap<>();
        ArrayList<Classification_Rule> rules = new ArrayList<>();

        int m = attributes.size();
        for (int k = 1; k <= m; k++) {
            List<List<Attribute>> attrCombinations = generateAttributeCombinations(attributes, k);

            for (List<Attribute> attrCombination : attrCombinations) {
                ArrayList<Attribute> attrCombinationArrayList = new ArrayList<>(attrCombination);
                Map<String, ArrayList<Integer>> valueGroups = groupInstancesByAttributeValues(inst, attrCombinationArrayList);

                for (Map.Entry<String, ArrayList<Integer>> entry : valueGroups.entrySet()) {
                    ArrayList<Integer> instanceSubset = entry.getValue();
                    
                    if (instanceSubset.size() <= 1) {
                        continue;
                    }
                    
                    Set<Integer> instanceSet = new HashSet<>(instanceSubset);
                    ArrayList<String> closure = computeAttributeClosure(inst, instanceSubset);

                    Classification_Rule rule = createClassificationRule(inst, instanceSubset, closure, attributes);
                    updateBestConcepts(bestConcepts, instanceSet, rule, closure);
                }
            }
        }

        // Construire la liste finale
        buildFinalConceptsList(inst, bestConcepts, allConcepts, rules);
        
        return rules;
    }
    
    // Méthode pour calculer la fermeture des attributs 
    private ArrayList<String> computeAttributeClosure(Instances inst, ArrayList<Integer> instances) {
        ArrayList<String> attributes = new ArrayList<>();
        
        if (instances.isEmpty()) {
            for (int i = 0; i < inst.numAttributes() - 1; i++) {
                attributes.add("-null-");
            }
            return attributes;
        }
        
        // Pour chaque attribut, vérifier si toutes les instances ont la même valeur
        for (int i = 0; i < inst.numAttributes() - 1; i++) {
            String firstValue = inst.instance(instances.get(0)).stringValue(i);
            boolean allSame = true;
            
            for (int j = 1; j < instances.size(); j++) {
                if (!inst.instance(instances.get(j)).stringValue(i).equals(firstValue)) {
                    allSame = false;
                    break;
                }
            }
            
            if (allSame) {
                attributes.add(firstValue);
            } else {
                attributes.add("-null-");
            }
        }
        
        return attributes;
    }

    
    private List<List<Attribute>> generateAttributeCombinations(ArrayList<Attribute> attributes, int k) {
	    List<List<Attribute>> combinations = new ArrayList<>();
	    generateCombinationsHelper(attributes, k, 0, new ArrayList<>(), combinations);
	    return combinations;
	}

    private void generateCombinationsHelper(List<Attribute> attributes, int k, int start, 
            List<Attribute> current, List<List<Attribute>> result) {
        if (current.size() == k) {
            result.add(new ArrayList<>(current));
            return;
        }
        
        for (int i = start; i < attributes.size(); i++) {
            current.add(attributes.get(i));
            generateCombinationsHelper(attributes, k, i + 1, current, result);
            current.remove(current.size() - 1);
        }
    }

 

    
    private Map<String, ArrayList<Integer>> groupInstancesByAttributeValues(Instances inst, 
            ArrayList<Attribute> attributes) {
        
        Map<String, ArrayList<Integer>> groups = new HashMap<>();
        
        for (int i = 0; i < inst.numInstances(); i++) {
            StringBuilder keyBuilder = new StringBuilder();
            
            for (Attribute attr : attributes) {
                keyBuilder.append(inst.instance(i).stringValue(attr))
                         .append(";");
            }
            
            String key = keyBuilder.toString();
            groups.computeIfAbsent(key, k -> new ArrayList<>()).add(i);
        }
        
        return groups;
    }
    
    /**
     * Crée une règle de classification
     */
    private Classification_Rule createClassificationRule(Instances inst, 
                                                      ArrayList<Integer> instanceSubset,
                                                      ArrayList<String> closure,
                                                      ArrayList<Attribute> attributes) throws Exception {
        int majorityClassIndex = findMajorityClass(inst, instanceSubset);
        Classification_Rule rule = new Classification_Rule(
            closure,
            majorityClassIndex,
            inst.classAttribute().value(majorityClassIndex)
        );

        ArrayList<String> attributeNames = new ArrayList<>();
        for (Attribute attr : attributes) {
            attributeNames.add(attr.name());
        }
        rule.setAttributeNames(attributeNames);
        rule.setRule_Ponderation(calculateRuleWeight(inst, rule));
        
        return rule;
    }
    
    /**
     * Calcule la pondération d'une règle (pourcentage d'instances couvertes)
     * @throws Exception 
     */
    private double calculateRuleWeight(Instances inst, Classification_Rule rule) throws Exception {
        double weight = 0;
        Classify_Instance classifier = new Classify_Instance();
        
        for (int i = 0; i < inst.numInstances(); i++) {
            if (classifier.TestClassifInstance_nom(inst.instance(i), rule) != false) {
                weight++;
            }
        }
        
        return weight / inst.numInstances();
    }
    /**
     * Trouve la classe majoritaire pour un ensemble d'instances
     */
    private int findMajorityClass(Instances inst, ArrayList<Integer> instances) {
        int[] classCounts = new int[inst.numClasses()];
        
        for (int i : instances) {
            int classIndex = (int) inst.instance(i).classValue();
            classCounts[classIndex]++;
        }
        
        int majorityClassIndex = 0;
        for (int i = 0; i < classCounts.length; i++) {
            if (classCounts[i] > classCounts[majorityClassIndex]) {
                majorityClassIndex = i;
            }
        }
        
        return majorityClassIndex;
    }

    /**
     * Met à jour les meilleurs concepts
     */
    private void updateBestConcepts(Map<Set<Integer>, Classification_Rule> bestConcepts,
                                 Set<Integer> instanceSet,
                                 Classification_Rule rule,
                                 ArrayList<String> closure) {
        if (bestConcepts.containsKey(instanceSet)) {
            Classification_Rule existingRule = bestConcepts.get(instanceSet);
            if (countValidAttributes(closure) > countValidAttributes(existingRule.getPremisses())) {
                bestConcepts.put(instanceSet, rule);
            }
        } else {
            bestConcepts.put(instanceSet, rule);
        }
    }

    /**
     * Construit la liste finale des concepts
     */
    private void buildFinalConceptsList(Instances inst,
                                     Map<Set<Integer>, Classification_Rule> bestConcepts,
                                     ArrayList<String> allConcepts,
                                     ArrayList<Classification_Rule> rules) {
        int conceptNum = 1;
        for (Map.Entry<Set<Integer>, Classification_Rule> entry : bestConcepts.entrySet()) {
            Set<Integer> instanceSet = entry.getKey();
            if (instanceSet.size() <= 1) {
                continue;
            }
            
            Classification_Rule rule = entry.getValue();
            String conceptStr = buildConceptString(inst, conceptNum++, instanceSet, rule);
            allConcepts.add(conceptStr);
            rules.add(rule);
        }
    }
    /**
     * Construit la représentation textuelle d'un concept
     */
    private String buildConceptString(Instances inst, int conceptNum, 
                                   Set<Integer> instanceSet, Classification_Rule rule) {
        StringBuilder conceptBuilder = new StringBuilder();
        conceptBuilder.append("Concept ").append(conceptNum).append(": {\n");
        
        // Ajout des objets
        conceptBuilder.append("  Objects: [\n");
        List<Integer> sortedInstances = new ArrayList<>(instanceSet);
        Collections.sort(sortedInstances);
        
        for (int i = 0; i < sortedInstances.size(); i++) {
            int objIndex = sortedInstances.get(i);
            conceptBuilder.append("    o").append(objIndex + 1).append(": {");
            
            for (int attrIdx = 0; attrIdx < inst.numAttributes(); attrIdx++) {
                if (attrIdx > 0) conceptBuilder.append(", ");
                Attribute attr = inst.attribute(attrIdx);
                String value = inst.instance(objIndex).stringValue(attrIdx);
                conceptBuilder.append(attr.name()).append("=").append(value);
            }
            conceptBuilder.append("}\n");
        }
        conceptBuilder.append("  ],\n");
        
        // Ajout des attributs
        conceptBuilder.append("  Attributes: [");
        boolean firstAttr = true;
        for (String attrValue : rule.getPremisses()) {
            if (!(attrValue.equals("-null-") || attrValue.equals("0") || attrValue.equals("0.0"))) {
                if (!firstAttr) conceptBuilder.append(", ");
                conceptBuilder.append(attrValue);
                firstAttr = false;
            }
        }
        conceptBuilder.append("]\n}");
        
        return conceptBuilder.toString();
    }

    /**
     * Affiche les concepts générés
     */
    public void printConcepts(ArrayList<String> allConcepts, Instances inst) {
        if (m_Debug && !allConcepts.isEmpty()) {
            System.out.println("\n=== GENERATED CONCEPTS LIST ===");
            System.out.println("Total number of concepts: " + allConcepts.size());
            System.out.println("--------------------------------------------");
            
            for (int i = 0; i < allConcepts.size(); i++) {
                System.out.println((i + 1) + ". " + allConcepts.get(i));
                if (i < allConcepts.size() - 1) {
                    System.out.println("--------------------------------------------");
                }
            }
            
            System.out.println("=== CONCEPTS LIST END===");
            
            afficheConceptLevels(allConcepts, inst);
        }
    }
        /**
         * Affiche les concepts groupés par niveau (nombre d'attributs)
         * @param allConcepts La liste complète des concepts à organiser par niveau
      
         * @param inst Les instances WEKA pour obtenir les noms des objets
         */
    private void afficheConceptLevels(ArrayList<String> allConcepts, Instances inst) {
            // Map pour organiser les concepts par niveau (nombre d'attributs)
            Map<Integer, List<String>> conceptsByLevel = new TreeMap<>();
            
            // Pattern pour extraire les objets et attributs des concepts
            Pattern objectPattern = Pattern.compile("Objects: \\[\\n(.*?)\\n  \\],", Pattern.DOTALL);
            Pattern attrPattern = Pattern.compile("Attributes: \\[(.*?)\\]");
            
            for (String concept : allConcepts) {
                // Extraire les objets
                Matcher objectMatcher = objectPattern.matcher(concept);
                List<String> objects = new ArrayList<>();
                if (objectMatcher.find()) {
                    String[] objLines = objectMatcher.group(1).split("\\n");
                    for (String line : objLines) {
                        objects.add(line.trim().replaceAll("o(\\d+):.*", "$1"));
                    }
                }
                
                // Extraire les attributs
                Matcher attrMatcher = attrPattern.matcher(concept);
                List<String> attributes = new ArrayList<>();
                if (attrMatcher.find()) {
                    String[] attrs = attrMatcher.group(1).split(",");
                    for (String attr : attrs) {
                        String trimmed = attr.trim();
                        if (!trimmed.isEmpty() && !trimmed.equals("-null-") 
                            && !trimmed.equals("0") && !trimmed.equals("0.0")) {
                            attributes.add(trimmed);
                        }
                    }
                }
                
                // Créer la représentation concise
                String conciseConcept = String.format("{[%s],[%s]}", 
                    String.join(",", objects), 
                    String.join(",", attributes));
                
                // Ajouter au niveau correspondant
                int level = attributes.size();
                conceptsByLevel.computeIfAbsent(level, k -> new ArrayList<>()).add(conciseConcept);
            }
            
            // Afficher les concepts par niveau
            System.out.println("\n=== CONCEPTS PER LEVEL ===");
            for (Map.Entry<Integer, List<String>> entry : conceptsByLevel.entrySet()) {
                System.out.printf("\n--- LEVEL %d (%d concepts) ---\n", 
                                 entry.getKey(), entry.getValue().size());
                
                // Afficher 5 concepts par ligne pour une meilleure lisibilité
                int count = 0;
                for (String concept : entry.getValue()) {
                    System.out.print(concept + "  ");
                    if (++count % 5 == 0) System.out.println();
                }
                if (count % 5 != 0) System.out.println();
            }
            System.out.println("=== END ===");
        }
        
    /**
     * Compte le nombre d'attributs valides dans une fermeture
     */
    private int countValidAttributes(ArrayList<String> closure) {
        int count = 0;
        for (String value : closure) {
            if (!(value.equals("-null-") || value.equals("0") || value.equals("0.0"))) {
                count++;
            }
        }
        return count;
    }
      


    /**
     * Génère les concepts formels à partir des valeurs majoritaires des attributs
     */
    public ArrayList<Classification_Rule> generateConceptsFromMajorityValues(Instances inst, 
                                                                            ArrayList<Attribute> attributes,
                                                                            Map<Attribute, String> majorityValues,
                                                                            ArrayList<String> allConcepts) throws Exception {
        Map<Set<Integer>, Classification_Rule> bestConcepts = new HashMap<>();
        ArrayList<Classification_Rule> rules = new ArrayList<>();

        // Générer toutes les combinaisons d'attributs de taille 1 à m
        int m = attributes.size();
        for (int k = 1; k <= m; k++) {
            List<List<Attribute>> attrCombinations = generateAttributeCombinations(attributes, k);

            for (List<Attribute> attrCombination : attrCombinations) {
                ArrayList<Attribute> attrCombinationArrayList = new ArrayList<>(attrCombination);
                
                // Trouver les instances ayant les valeurs majoritaires pour ces attributs
                ArrayList<Integer> majorityInstances = findInstancesWithMajorityValues(inst, 
                    attrCombinationArrayList, majorityValues);

                if (majorityInstances.isEmpty()) continue;

                Set<Integer> instanceSet = new HashSet<>(majorityInstances);
                
                // Calculer la fermeture pour ces attributs
                ArrayList<String> closure = computeMultiAttributeClosure(inst, 
                    majorityInstances, attrCombinationArrayList);

                // Créer la règle de classification
                Classification_Rule rule = createMajorityValueRule(inst, majorityInstances, 
                    closure, attributes, attrCombinationArrayList, majorityValues);

                // Mettre à jour les meilleurs concepts
                updateBestConcepts(bestConcepts, instanceSet, rule, closure);
            }
        }

        // Construire la liste finale des concepts
        buildFinalConceptsList(inst, bestConcepts, allConcepts, rules);

        return rules;
    }

    
    
    /**
     * Crée une règle à partir des valeurs majoritaires
     */
    private Classification_Rule createMajorityValueRule(Instances inst, 
                                                      ArrayList<Integer> majorityInstances,
                                                      ArrayList<String> closure,
                                                      ArrayList<Attribute> allAttributes,
                                                      ArrayList<Attribute> combinationAttributes,
                                                      Map<Attribute, String> majorityValues) throws Exception {
        // Trouver la classe majoritaire
        int majorityClassIndex = findMajorityClass(inst, majorityInstances);

        // Créer la règle
        Classification_Rule rule = new Classification_Rule(
            closure,
            majorityClassIndex,
            inst.classAttribute().value(majorityClassIndex)
        );

        // Ajouter les noms d'attributs
        ArrayList<String> attributeNames = new ArrayList<>();
        for (Attribute attr : allAttributes) {
            attributeNames.add(attr.name());
        }
        rule.setAttributeNames(attributeNames);

        // Calculer le poids
        double weight = calculateRuleWeight(inst, rule);
        rule.setRule_Ponderation(weight);

        return rule;
    }

    /**
     * Trouve les instances ayant les valeurs majoritaires pour une combinaison d'attributs
     */
    private ArrayList<Integer> findInstancesWithMajorityValues(Instances inst, 
                                                             ArrayList<Attribute> attributes,
                                                             Map<Attribute, String> majorityValues) {
        ArrayList<Integer> instances = new ArrayList<>();

        for (int i = 0; i < inst.numInstances(); i++) {
            boolean allMajority = true;
            for (Attribute attr : attributes) {
                String currentValue = inst.instance(i).stringValue(attr);
                if (!currentValue.equals(majorityValues.get(attr))) {
                    allMajority = false;
                    break;
                }
            }
            if (allMajority) {
                instances.add(i);
            }
        }

        return instances;
    }

    /**
     * Calcule la fermeture pour plusieurs attributs avec valeurs majoritaires
     */
    private ArrayList<String> computeMultiAttributeClosure(Instances inst, 
                                                         ArrayList<Integer> instances,
                                                         ArrayList<Attribute> attributes) {
        ArrayList<String> closure = new ArrayList<>();
        
        // Pour chaque attribut dans la combinaison, ajouter sa valeur majoritaire
        // Pour les autres attributs, mettre "-null-"
        for (int i = 0; i < inst.numAttributes() - 1; i++) {
            Attribute currentAttr = inst.attribute(i);
            if (attributes.contains(currentAttr)) {
                // Trouver la valeur majoritaire pour cet attribut
                String majorityValue = findMajorityValue(inst, i);
                closure.add(majorityValue);
            } else {
                closure.add("-null-");
            }
        }
        
        return closure;
    }

    /**
     * Trouve la valeur majoritaire pour un attribut
     */
    private String findMajorityValue(Instances inst, int attrIndex) {
        Map<String, Integer> valueCounts = new HashMap<>();
        
        for (int i = 0; i < inst.numInstances(); i++) {
            String value = inst.instance(i).stringValue(attrIndex);
            valueCounts.put(value, valueCounts.getOrDefault(value, 0) + 1);
        }
        
        String majorityValue = null;
        int maxCount = -1;
        
        for (Map.Entry<String, Integer> entry : valueCounts.entrySet()) {
            if (entry.getValue() > maxCount) {
                maxCount = entry.getValue();
                majorityValue = entry.getKey();
            }
        }
        
        return majorityValue;
    }
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
}