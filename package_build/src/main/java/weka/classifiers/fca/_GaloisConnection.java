package weka.classifiers.fca;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Set;
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
     * Generate concepts using the Galois connection
     * 
     * @return List of formal concepts (pairs of object sets and attribute sets)
     */
    
    public ArrayList<FormalConcept> generateConcepts() {
        ArrayList<FormalConcept> concepts = new ArrayList<>();
        
        // Generate all possible valid attribute combinations
        List<Set<String>> attributeCombinations = generateAllValidCombinations();
        
        // Process each combination to find formal concepts
        for (Set<String> intent : attributeCombinations) {
            Set<Integer> extent = getExtent(intent);
            Set<String> closedIntent = getIntent(extent);
            
            // Only keep if the intent is closed (equal to its closure)
            if (intent.equals(closedIntent)) {
                FormalConcept concept = new FormalConcept(extent, intent);
                
                // Check if this concept is already present
                if (!containsConcept(concepts, concept)) {
                    concepts.add(concept);
                }
            }
        }
        
        if (m_Debug) {
            System.out.println("Generated " + concepts.size() + " formal concepts");
            for (int i=0;i<concepts.size();i++) {
                System.out.println("concept"+i);
                System.out.println("extent"+concepts.get(i).getObjects());
                System.out.println("intent"+concepts.get(i).getAttributes());


            }
        }
        
        return concepts;
    }

    private List<Set<String>> generateAllValidCombinations() {
        List<Set<String>> combinations = new ArrayList<>();
        
        // First get all attributes with their possible values
        Map<String, List<String>> attributeValues = new LinkedHashMap<>();
        for (int j = 0; j < m_Instances.numAttributes(); j++) {
            if (j == m_Instances.classIndex()) continue;
            
            Attribute attr = m_Instances.attribute(j);
            List<String> values = new ArrayList<>();
            for (int k = 0; k < attr.numValues(); k++) {
                values.add(attr.name() + "=" + attr.value(k));
            }
            attributeValues.put(attr.name(), values);
        }
        
        // Generate all possible non-empty subsets
        List<String> attributes = new ArrayList<>(attributeValues.keySet());
        generateCombinations(attributes, 0, attributeValues, new HashSet<>(), combinations);
        
        return combinations;
    }

    private void generateCombinations(List<String> attributes, int index, 
                                    Map<String, List<String>> attributeValues,
                                    Set<String> current, 
                                    List<Set<String>> combinations) {
        if (index >= attributes.size()) {
            if (!current.isEmpty()) {
                combinations.add(new HashSet<>(current));
            }
            return;
        }
        
        // Option 1: Don't include this attribute
        generateCombinations(attributes, index+1, attributeValues, current, combinations);
        
        // Option 2: Include one value for this attribute
        String attrName = attributes.get(index);
        for (String value : attributeValues.get(attrName)) {
            Set<String> newCombination = new HashSet<>(current);
            newCombination.add(value);
            generateCombinations(attributes, index+1, attributeValues, newCombination, combinations);
        }
    }

    private boolean containsConcept(ArrayList<FormalConcept> concepts, FormalConcept newConcept) {
        for (FormalConcept concept : concepts) {
            if (concept.getObjects().equals(newConcept.getObjects()) && 
                concept.getAttributes().equals(newConcept.getAttributes())) {
                return true;
            }
        }
        return false;
    }
    
    
    
    /**
     * Generate all formal concepts by computing Galois closure from all possible combinations
     * of attributes and their values
     * 
     * @return List of formal concepts
     */
    public ArrayList<FormalConcept> generateConcepts2() {
        ArrayList<FormalConcept> concepts = new ArrayList<>();
        
        // 1. Préparer la liste de tous les attributs avec leurs valeurs
        List<Attribute> attributes = new ArrayList<>();
        for (int j = 0; j < m_Instances.numAttributes(); j++) {
            if (j != m_Instances.classIndex()) {
                attributes.add(m_Instances.attribute(j));
            }
        }
        
        // 2. Trouver le nombre maximum de valeurs pour un attribut
        int maxValues = 0;
        for (Attribute attr : attributes) {
            if (attr.numValues() > maxValues) {
                maxValues = attr.numValues();
            }
        }
        
        // 3. Générer toutes les combinaisons possibles
        int nbAttributes = attributes.size();
        long totalCombinations = (long) Math.pow(maxValues + 1, nbAttributes);
        
        for (long i = 1; i < totalCombinations; i++) {
            Set<String> intent = new HashSet<>();
            long code = i;
            
            // Décoder la combinaison
            for (int attrIndex = 0; attrIndex < nbAttributes; attrIndex++) {
                Attribute attr = attributes.get(attrIndex);
                int valueIndex = (int) (code % (maxValues + 1));
                code = code / (maxValues + 1);
                
                if (valueIndex < attr.numValues()) {
                    intent.add(attr.name() + "=" + attr.value(valueIndex));
                }
            }
            
            // Vérifier que l'intent est valide (pas de doublons d'attributs)
            if (isValidIntent(intent)) {
                // Calculer la fermeture de Galois
                Set<Integer> extent = getExtent(intent);
                Set<String> closedIntent = getIntent(extent);
                
                // Créer le concept si c'est une fermeture
                if (intent.equals(closedIntent)) {
                    FormalConcept concept = new FormalConcept(extent, closedIntent);
                    
                    // Vérifier si le concept existe déjà
                    if (!containsConcept(concepts, concept)) {
                        concepts.add(concept);
                    }
                }
            }
        }
        if (m_Debug) {
            System.out.println("Generated " + concepts.size() + " formal concepts");
            for (int i=0;i<concepts.size();i++) {
                System.out.println("concept"+i);
                System.out.println("extent"+concepts.get(i).getObjects());
                System.out.println("intent"+concepts.get(i).getAttributes());


            }
        }
        return concepts;
    
   }
    private boolean isValidIntent(Set<String> intent) {
        Set<String> attributeNames = new HashSet<>();
        for (String attrVal : intent) {
            String attrName = attrVal.split("=")[0];
            if (attributeNames.contains(attrName)) {
                return false; // Attribut en double
            }
            attributeNames.add(attrName);
        }
        return true;
    }

 
    
    
    private List<Set<String>> generateValidAttributeCombinations(List<Attribute> attributes) {
        List<Set<String>> combinations = new ArrayList<>();
        generateCombinationsRecursive(attributes, 0, new HashSet<>(), combinations);
        return combinations;
    }

    private void generateCombinationsRecursive(List<Attribute> attributes, int currentIndex, 
                                             Set<String> currentCombination,
                                             List<Set<String>> result) {
        if (currentIndex >= attributes.size()) {
            if (!currentCombination.isEmpty()) {
                result.add(new HashSet<>(currentCombination));
            }
            return;
        }
        
        Attribute currentAttr = attributes.get(currentIndex);
        
        // Option 1: Don't include this attribute
        generateCombinationsRecursive(attributes, currentIndex + 1, 
                                    currentCombination, result);
        
        // Option 2: Include one value for this attribute
        for (int i = 0; i < currentAttr.numValues(); i++) {
            Set<String> newCombination = new HashSet<>(currentCombination);
            newCombination.add(currentAttr.name() + "=" + currentAttr.value(i));
            generateCombinationsRecursive(attributes, currentIndex + 1,
                                        newCombination, result);
        }
    }

    private boolean conceptExists(ArrayList<FormalConcept> concepts, 
                                Set<Integer> extent, Set<String> intent) {
        for (FormalConcept concept : concepts) {
            if (concept.getObjects().equals(extent) && 
                concept.getAttributes().equals(intent)) {
                return true;
            }
        }
        return false;
    }
    
    
   
    
    /**
     * Helper method to generate all valid combinations of attribute-value pairs
     * where each attribute appears at most once
     */
    private List<Set<String>> generateValidCombinations(Map<String, Set<String>> attributesByGroup) {
        List<Set<String>> result = new ArrayList<>();
        List<String> attributeNames = new ArrayList<>(attributesByGroup.keySet());
        
        generateCombinationsRecursive(attributesByGroup, attributeNames, 0, new HashSet<>(), result);
        
        return result;
    }

    /**
     * Recursive helper to generate attribute combinations
     */
    private void generateCombinationsRecursive(Map<String, Set<String>> attributesByGroup, 
                                             List<String> attributeNames, 
                                             int currentIndex, 
                                             Set<String> currentCombination,
                                             List<Set<String>> result) {
        if (currentIndex >= attributeNames.size()) {
            if (!currentCombination.isEmpty()) {
                result.add(new HashSet<>(currentCombination));
            }
            return;
        }
        
        String currentAttr = attributeNames.get(currentIndex);
        
        // Option 1: Don't include any value for this attribute
        generateCombinationsRecursive(attributesByGroup, attributeNames, currentIndex + 1, 
                                    currentCombination, result);
        
        // Option 2: Include one value for this attribute
        for (String value : attributesByGroup.get(currentAttr)) {
            Set<String> newCombination = new HashSet<>(currentCombination);
            newCombination.add(currentAttr + "=" + value.split("=")[1]);
            generateCombinationsRecursive(attributesByGroup, attributeNames, currentIndex + 1,
                                        newCombination, result);
        }
    }

    
    
       
  
    
   
    /**
     * Recursive helper for concept generation
     * This is a simplified version of the NextClosure algorithm
     * 
     * @param concepts List to store the generated concepts
     * @param currentConcept The current concept to expand from
     */
    private void generateConceptsRecursive(ArrayList<FormalConcept> concepts, FormalConcept currentConcept) {
        // Try adding each object and see if we get a new concept
        for (int i = 0; i < m_Instances.numInstances(); i++) {
            if (currentConcept.getObjects().contains(i)) continue;
            
            // Create new object set with added object
            Set<Integer> newObjects = new HashSet<>(currentConcept.getObjects());
            newObjects.add(i);
            
            // Apply Galois connection
            Set<String> newAttributes = getIntent(newObjects);
            Set<Integer> closedObjects = getExtent(newAttributes);
            
            // Create new concept
            FormalConcept newConcept = new FormalConcept(closedObjects, newAttributes);
            
            // Check if this concept is already in the list
            boolean exists = false;
            for (FormalConcept concept : concepts) {
                if (concept.equals(newConcept)) {
                    exists = true;
                    break;
                }
            }
            
            if (!exists) {
                concepts.add(newConcept);
                generateConceptsRecursive(concepts, newConcept);
            }
        }
    }
    
    /**
     * Generate classification rules from formal concepts
     * 
     * @return List of classification rules
     */
    public ArrayList<Classification_Rule> generateRules() {
        ArrayList<Classification_Rule> rules = new ArrayList<>();
        
        // Generate concepts
        ArrayList<FormalConcept> concepts = generateConcepts();
        
        // For each concept, create a rule
        for (FormalConcept concept : concepts) {
            
            // Skip concepts with empty extent (no objects)
            if (concept.getObjects().isEmpty()) continue;
            
            // Find majority class for this concept
            int majorityClass = findMajorityClass(concept.getObjects());
            
            // Convert attribute set to ArrayList<String> format used by Classification_Rule
            ArrayList<String> attributes = convertAttributesToRuleFormat(concept.getAttributes());
            
            // Create rule
            Classification_Rule rule = new Classification_Rule(
                attributes,
                majorityClass,
                m_Instances.attribute(m_Instances.classIndex()).value(majorityClass)
            );
            
            // Calculate rule weight
            double weight = calculateRuleWeight(rule);
            rule.setRule_Ponderation(weight);
            
            rules.add(rule);
        }
        
        if (m_Debug) {
            System.out.println("Generated " + rules.size() + " classification rules from concepts");
        }
        
        return rules;
    }
    
    /**
     * Generate rules using a specific Galois connection strategy
     * 
     * @param strategy The Galois connection strategy to use
     * @return List of classification rules
     * @throws Exception if rule generation fails
     */
    public ArrayList<Classification_Rule> generateRulesWithStrategy(int strategy) throws Exception {
        if (m_Debug) {
            System.out.println("Generating rules with Galois strategy: " + strategy);
        }
        
        switch (strategy) {
            case GALOIS_PERTINENT_MAJORITY:
                return generateRulesPertinentMajority();
            case GALOIS_PERTINENT_ALL:
                return generateRulesPertinentAll();
            case GALOIS_ALL_MAJORITY:
                return generateRulesAllMajority();
            case GALOIS_ALL_ALL:
                return generateRulesAllAll();
            default:
                throw new IllegalArgumentException("Unknown Galois strategy: " + strategy);
        }
    }
    
    /**
     * Generate rules using pertinent attribute and majority value
     * 
     * @return List of classification rules
     * @throws Exception if rule generation fails
     */
    private ArrayList<Classification_Rule> generateRulesPertinentMajority() throws Exception {
        // TO FIX
        return generateRules();
    }
    
    /**
     * Generate rules using pertinent attribute and all values
     * 
     * @return List of classification rules
     * @throws Exception if rule generation fails
     */
    private ArrayList<Classification_Rule> generateRulesPertinentAll() throws Exception {
        // TO FIX
        return generateRules();
    }
    
    /**
     * Generate rules using all attributes and majority values
     * 
     * @return List of classification rules
     * @throws Exception if rule generation fails
     */
    private ArrayList<Classification_Rule> generateRulesAllMajority() throws Exception {
        // TO FIX
        return generateRules();
    }
    
    /**
     * Generate rules using all attributes and all values
     * 
     * @return List of classification rules
     * @throws Exception if rule generation fails
     */
    private ArrayList<Classification_Rule> generateRulesAllAll() throws Exception {
        // This uses all attributes and all their values
        // to generate rules - full concept lattice
        return generateRules();
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
}