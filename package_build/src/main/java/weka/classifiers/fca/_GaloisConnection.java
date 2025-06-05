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
import weka.core.Instance;
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
    
    
    public static final int POLICY_FIRST = 1;
    public static final int POLICY_LAST = 2;
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
     * Class representing a formal concept (pair of object set and attribute set)
     */
    public class FormalConcept {
        private Set<Integer> objects;
        private Set<String> attributes;
        
        public FormalConcept() {
        	
        }
        
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
            StringBuilder sb = new StringBuilder();
            
            sb.append("{");
            sb.append(
                objects.stream()
                       .map(obj -> "O" + obj)  
                       .collect(Collectors.joining(",", "[", "]")) // Format [O3,O4,O9]
            );
            
            // Formatage des attributs (sans modification)
            sb.append(",");
            sb.append(attributes.toString().replace(" ", ""));
            
            sb.append("}");
            return sb.toString();
        }
    }
    

    /**
     * Vérifie si un dataset est binaire (tous les attributs ont des valeurs 0/1 ou false/true)
     * @param inst Le dataset Weka à vérifier
     * @return true si le dataset est binaire, false sinon
     */
    public static boolean isBinaryDataset(Instances inst) {
        for (int i = 0; i < inst.numAttributes(); i++) {
            Attribute attr = inst.attribute(i);
            
            // Ignorer l'attribut classe si nécessaire
            if (i == inst.classIndex()) {
                continue;
            }
            
            // Vérifier le type d'attribut
            if (attr.isNumeric()) {
                // Pour les attributs numériques, vérifier les valeurs 0/1
                if (!isBinaryNumericAttribute(inst, i)) {
                    return false;
                }
            } else if (attr.isNominal()) {
                // Pour les attributs nominaux, vérifier les valeurs false/true ou 0/1
                if (!isBinaryNominalAttribute(attr)) {
                    return false;
                }
            } else {
                // Autres types (date, string...) ne sont pas binaires
                return false;
            }
        }
        return true;
    }

    /**
     * Vérifie si un attribut numérique ne contient que 0 et 1
     */
    private static boolean isBinaryNumericAttribute(Instances inst, int attrIndex) {
        for (int i = 0; i < inst.numInstances(); i++) {
            double value = inst.instance(i).value(attrIndex);
            if (value != 0.0 && value != 1.0) {
                return false;
            }
        }
        return true;
    }

    /**
     * Vérifie si un attribut nominal a exactement 2 valeurs (false/true ou 0/1)
     */
    private static boolean isBinaryNominalAttribute(Attribute attr) {
        if (attr.numValues() != 2) {
            return false;
        }
        
        // Vérifie si les valeurs sont false/true ou 0/1
        String val1 = attr.value(0).toLowerCase();
        String val2 = attr.value(1).toLowerCase();
        
        return (val1.equals("false") && val2.equals("true")) ||
               (val1.equals("true") && val2.equals("false")) ||
               (val1.equals("0") && val2.equals("1")) ||
               (val1.equals("1") && val2.equals("0"));
    }
    
    /**
     * Génère les concepts formels à partir des instances et attributs
     */
    ArrayList<Classification_Rule> generateConceptsAllCombinations(Instances inst, 
                                                         ArrayList<Attribute> attributes,
                                                         ArrayList<String> allConcepts,
                                                         boolean debug) throws Exception {
        Map<Set<Integer>, Classification_Rule> bestConcepts = new HashMap<>();
        ArrayList<Classification_Rule> rules = new ArrayList<>();
        
        System.out.println("generateConcepts:: Data set binary: "+isBinaryDataset(inst));

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
        List<FormalConcept> conceptsList = buildFinalConceptsList(inst, bestConcepts, allConcepts, rules);
        if (debug) {
        	String con= getConceptsAsString(conceptsList,inst);
        	System.out.println("TEST AFFICH::::::::::");
        	System.out.println(con);
        	afficheConceptLevels(conceptsList);
        }
        
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

    
    /**
     * Génère les concepts formels en considérant tous les attributs individuellement (sans combinaisons)
     * tout en réutilisant la même structure que generateConceptsAllCombinations
     */
    public ArrayList<Classification_Rule> generateConcepts(Instances inst, 
                                                         ArrayList<Attribute> attributes,
                                                         ArrayList<String> allConcepts,
                                                         boolean debug) throws Exception {
        Map<Set<Integer>, Classification_Rule> bestConcepts = new HashMap<>();
        ArrayList<Classification_Rule> rules = new ArrayList<>();

        System.out.println("generateConcepts:: Data set binary: "+isBinaryDataset(inst));

        // Version simplifiée qui ne génère pas de combinaisons
        for (Attribute attr : attributes) {
            ArrayList<Attribute> singleAttrList = new ArrayList<>();
            singleAttrList.add(attr);

            // Grouper les instances par valeurs de cet attribut unique
            Map<String, ArrayList<Integer>> valueGroups = groupInstancesByAttributeValues(inst, singleAttrList);

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

        // Construire la liste finale (même méthode que generateConceptsAllCombinations)
        List<FormalConcept> conceptsList = buildFinalConceptsList(inst, bestConcepts, allConcepts, rules);
        if (debug) {
            printConcepts(conceptsList, inst);
        }
        
        return rules;
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
    private List<FormalConcept> buildFinalConceptsList(Instances inst,
                                     Map<Set<Integer>, Classification_Rule> bestConcepts,
                                     ArrayList<String> allConcepts,
                                     ArrayList<Classification_Rule> rules) {
        List<FormalConcept> concepts =new ArrayList<>();
        for (Map.Entry<Set<Integer>, Classification_Rule> entry : bestConcepts.entrySet()) {
            Set<Integer> instanceSet = entry.getKey();
            if (instanceSet.size() <= 1) {
                continue;
            }
            
            Classification_Rule rule = entry.getValue();
            FormalConcept concept=buildConcept(instanceSet, rule);
            //pour l'export de concepts
            String conceptStr= concept.toString();
            concepts.add(concept);
            
            
            allConcepts.add(conceptStr);
            rules.add(rule);
        }
        
        String conceptsString=  JsonExporter.toJson(concepts);

        String path="C://Users//asus//Desktop//generated_concepts.json";
        JsonExporter.saveToFile(conceptsString, path);
		return concepts;
    }
  
   
    /**
     * Retourne une chaîne de caractères représentant la liste de concepts formattée.
     */
    public String getConceptsAsString(List<FormalConcept> concepts, Instances inst) {
        StringBuilder sb = new StringBuilder();
        int count = 1;

        for (FormalConcept concept : concepts) {
            sb.append("--------------------------------------------\n\n");

            sb.append(count).append(". Concept:\n");

            sb.append("Objects (instances):\n");
            for (Integer objIndex : concept.getObjects()) {
                Instance instance = inst.instance(objIndex);
                sb.append("  o").append(objIndex + 1).append(" : ");
                List<String> attrValues = new ArrayList<>();

                for (int i = 0; i < inst.numAttributes(); i++) {
                    Attribute attr = inst.attribute(i);
                    String value = instance.toString(attr);
                    attrValues.add(attr.name() + "=" + value);
                }

                sb.append(String.join(", ", attrValues)).append("\n");
            }

            sb.append("\nConnexion Attributes:\n");
            for (String attr : concept.getAttributes()) {
                sb.append("  - ").append(attr).append("\n");
            }

            sb.append("\n");
            count++;
        }

        sb.append("--------------------------------------------\n");
        return sb.toString();
    }

    
    /**
     * Affiche les concepts générés à partir d'une liste d'objets FormalConcept
     */
    public void printConcepts(List<FormalConcept> allConcepts, Instances inst) {
        if (m_Debug && allConcepts != null && !allConcepts.isEmpty()) {
            System.out.println("\n=== GENERATED CONCEPTS LIST ===");
            System.out.println("Total number of concepts: " + allConcepts.size());
            
            for (int i = 0; i < allConcepts.size(); i++) {
                FormalConcept concept = allConcepts.get(i);
                System.out.println("\n" + (i + 1) + ". Concept:");
                
                // Affichage des instances
                System.out.println("Objects (instances):");
                concept.getObjects().stream()
                      .sorted()
                      .forEach(idx -> printInstance(inst, idx));
                
                // Affichage des attributs
                System.out.println("\nConnexion Attributes:");
                concept.getAttributes().stream()
                      .sorted()
                      .forEach(attr -> System.out.println("  - " + attr));
                
                if (i < allConcepts.size() - 1) {
                    System.out.println("\n--------------------------------------------");
                }
            }
            System.out.println("\n=== CONCEPTS LIST END ===");
        }
        afficheConceptLevels(allConcepts);
    }

    private void printInstance(Instances inst, int index) {
        Instance instance = inst.instance(index);
        StringBuilder sb = new StringBuilder("  o" + (index+1) + " : ");
        for (int j = 0; j < inst.numAttributes(); j++) {
            Attribute attr = inst.attribute(j);
            sb.append(attr.name()).append("=");
            if (attr.isNominal()) {
                sb.append(instance.stringValue(j));
            } else {
                sb.append(instance.value(j));
            }
            if (j < inst.numAttributes() - 1) sb.append(", ");
        }
        System.out.println(sb.toString());
    }
    
    /**
     * Affiche les concepts groupés par niveau (nombre d'attributs)
     * @param allConcepts La liste complète des concepts
     */
    public void afficheConceptLevels(List<FormalConcept> allConcepts) {
        // Map pour organiser les concepts par niveau (nombre d'attributs)
        Map<Integer, List<String>> conceptsByLevel = new TreeMap<>();

        for (FormalConcept concept : allConcepts) {
            int level = concept.getAttributes().size();

            // Création de la représentation concise
            String objStr = concept.getObjects().stream()
                .map(i -> "O" + i)
                .collect(Collectors.joining(","));

            String attrStr = concept.getAttributes().stream()
                .collect(Collectors.joining(","));

            String concise = String.format("{[%s],[%s]}", objStr, attrStr);

            conceptsByLevel
                .computeIfAbsent(level, k -> new ArrayList<>())
                .add(concise);
        }

        // Affichage
        System.out.println("\n=== CONCEPTS BY INTENTION SET SIZE ===");
        for (Map.Entry<Integer, List<String>> entry : conceptsByLevel.entrySet()) {
            System.out.printf("\n--- SIZE %d (%d concepts) ---\n", 
                              entry.getKey(), entry.getValue().size());

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
    public ArrayList<Classification_Rule> generateConceptsFromMajorityValuesAllCombinations(Instances inst, 
                                                                            ArrayList<Attribute> attributes,
                                                                            Map<Attribute, String> majorityValues,
                                                                            ArrayList<String> allConcepts,
                                                                            boolean debug) throws Exception {
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
        List<FormalConcept> conceptsList =  buildFinalConceptsList(inst, bestConcepts, allConcepts, rules);

        if (debug) {
        	printConcepts(conceptsList,inst);
        }
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
	    
	    
	    /**
	     * Génère des règles pour chaque attribut avec sa valeur majoritaire
	     */
	    public ArrayList<Classification_Rule> generateConceptsFromMajorityValues(Instances inst, 
	                                                                         ArrayList<Attribute> attributes,
	                                                                         ArrayList<String> allConcepts,
	                                                                         int Policy_Selection,
	                                                                         boolean debug) throws Exception {
	        ArrayList<Classification_Rule> rules = new ArrayList<>();
	        Map<Set<Integer>, Classification_Rule> bestConcepts = new HashMap<>();

	        // 1. Pour chaque attribut individuellement
	        for (Attribute attr : attributes) {
	            // 2. Trouver la valeur majoritaire de cet attribut
	            int majorityIdx = findMajorityValueIndex(inst, attr.index(), Policy_Selection);
	            String majorityValue = attr.value(majorityIdx);

	            // 3. Trouver les instances ayant cette valeur majoritaire
	            ArrayList<Integer> matchingInstances = new ArrayList<>();
	            for (int i = 0; i < inst.numInstances(); i++) {
	                if (inst.instance(i).stringValue(attr).equals(majorityValue)) {
	                    matchingInstances.add(i);
	                }
	            }

	            if (matchingInstances.size() > 1) { // Seuil minimal d'instances
	                // 4. Calculer la fermeture pour ces instances
	                ArrayList<String> closure = computeAttributeClosure(inst, matchingInstances);

	                // 5. Créer une règle spécifique à cet attribut
	                Classification_Rule rule = createClassificationRule(inst, matchingInstances, closure, attributes);
	                
	                // 6. Mettre à jour les meilleurs concepts
	                Set<Integer> instanceSet = new HashSet<>(matchingInstances);
	                updateBestConcepts(bestConcepts, instanceSet, rule, closure);
	            }
	        }

	        // 7. Construire le résultat final
	        List<FormalConcept> concepts =buildFinalConceptsList(inst, bestConcepts, allConcepts, rules);
	        if (debug) {
	            printConcepts(concepts, inst);
	        }
	        
	        return rules;
	    }
	    
	    

    /**
     * Trouve l'index de la valeur majoritaire d'un attribut
     * @param inst Dataset complet
     * @param attrIdx Index de l'attribut à analyser
     * @return Index de la valeur majoritaire (selon Policy_Selection en cas d'égalité)
     */
    private int findMajorityValueIndex(Instances inst, int attrIdx,int Policy_Selection) {
        Attribute attribute = inst.attribute(attrIdx);
        int[] valueCounts = new int[attribute.numValues()];
        
        // Compter les occurrences de chaque valeur
        for (int i = 0; i < inst.numInstances(); i++) {
            int valueIdx = (int) inst.instance(i).value(attrIdx);
            valueCounts[valueIdx]++;
        }
        
        // Trouver la valeur majoritaire
        int majorityValueIdx = 0;
        for (int i = 0; i < valueCounts.length; i++) {
            if (valueCounts[i] > valueCounts[majorityValueIdx]) {
                majorityValueIdx = i;
            } else if (valueCounts[i] == valueCounts[majorityValueIdx] && Policy_Selection == POLICY_LAST) {
                majorityValueIdx = i;
            }
        }
        
        return majorityValueIdx;
    }
    
    
    private FormalConcept buildConcept(Set<Integer> instanceSet, Classification_Rule rule) {
        Set<String> labeledAttributes = new HashSet<>();

        for (int i = 0; i < rule.getPremisses().size(); i++) {
            String attrValue = rule.getPremisses().get(i);

            if (!(attrValue.equals("-null-") || attrValue.equals("0") || attrValue.equals("0.0"))) {
                String attrName = rule.getAttributeNames().get(i);

                // Si attrValue contient déjà "=", ne le doubler pas
                if (attrValue.contains("=")) {
                    labeledAttributes.add(attrValue); // attrValue est déjà du type "attribute=value"
                } else {
                    labeledAttributes.add(attrName + "=" + attrValue);
                }
            }
        }

        return new FormalConcept(instanceSet, labeledAttributes);
    }

    
}