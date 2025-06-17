package weka.classifiers.fca;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Set;
import java.util.TreeMap;
import java.util.TreeSet;
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
    
    private List<FormalConcept> conceptList;
    // The formal context
    private Instances m_Instances;
    
    // Maps for the Galois connections
    private Map<Set<Integer>, Set<String>> intentMap;  // Object set -> Attribute set
    private Map<Set<String>, Set<Integer>> extentMap;  // Attribute set -> Object set
    
    
    public void setConceptList(List<FormalConcept> list){
    	this.conceptList=list;
    }
    
    public List<FormalConcept> getConceptList( ){
    	return this.conceptList;
    }
    /**
     * Constructor
     * 
     * @param debug Debug mode flag
     */
    public _GaloisConnection(boolean debug) {
        this.m_Debug = debug;
        this.intentMap = new HashMap<>();
        this.extentMap = new HashMap<>();
        this.conceptList= new ArrayList<>();
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
     * Construit un tableau aligné sur les attributs du dataset :
     * - si l’attribut binaire est présent dans le concept → "att=val"
     * - sinon → "-null-"
     *
     * @param binAttrNames   liste des noms d’attributs binaires (ex. outlook=sunny)
     * @param inst           dataset Weka
     */
    private ArrayList<String> convertAttributesToRuleFormat(ArrayList<String> binAttrNames,
                                                            Instances inst) {
        ArrayList<String> result = new ArrayList<>();

        // 1) initialiser à -null-
        for (int i = 0; i < inst.numAttributes() - 1; i++) {   // -1 pour la classe
            result.add("-null-");
        }

        // 2) pour chaque attribut binaire réellement présent dans la règle / concept
        for (String fullName : binAttrNames) {                 // fullName = "outlook=sunny"
            // trouver l’index exact de cet attribut dans le dataset
            Attribute attr = inst.attribute(fullName);
            if (attr == null) continue;                        // sécurité

            int pos = attr.index();
            if (pos == inst.classIndex()) continue;            // on saute la classe

            // position « compacte » (sans la classe)
            int compactPos = (pos < inst.classIndex()) ? pos : pos - 1;

            // on place la paire entière "outlook=sunny"
            result.set(compactPos, fullName);
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
    
    private ArrayList<Attribute> filterBinaryAttributes(ArrayList<Attribute> attributes, Instances inst) {
        ArrayList<Attribute> filtered = new ArrayList<>();

        for (Attribute attr : attributes) {
            boolean hasPositiveValue = false;

            for (int i = 0; i < inst.numInstances(); i++) {
                double val = inst.instance(i).value(attr);
                String valStr = attr.isNominal() ? inst.instance(i).stringValue(attr) : String.valueOf(val);

                if (!valStr.equals("0") && !valStr.equals("0.0") && !valStr.equalsIgnoreCase("false")) {
                    hasPositiveValue = true;
                    break;
                }
            }

            if (hasPositiveValue) {
                filtered.add(attr);
            }
        }

        return filtered;
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
        //afficheAttributsEtValeurs(inst);

        
        int m = attributes.size();
        for (int k = 1; k <= m; k++) {
            List<List<Attribute>> attrCombinations = generateAttributeCombinations(inst,attributes, k,debug);

            for (List<Attribute> attrCombination : attrCombinations) {
                ArrayList<Attribute> attrCombinationArrayList = new ArrayList<>(attrCombination);
                Map<String, ArrayList<Integer>> valueGroups = groupInstancesByAttributeValues(inst, attrCombinationArrayList);

                for (Map.Entry<String, ArrayList<Integer>> entry : valueGroups.entrySet()) {
                    ArrayList<Integer> instanceSubset = entry.getValue();
                    
                    if (instanceSubset.size() <= 1) {
                        continue;
                    }
                    
                    Set<Integer> instanceSet = new HashSet<>(instanceSubset);
                    Map<String,String> closure = computeAttributeClosure(inst, instanceSubset,attrCombinationArrayList);
                 // Parcours de tous les attributs
                    for (Map.Entry<String, String> c : closure.entrySet()) {
                        System.out.println(c.getKey() + ": " + c.getValue());
                    }
                    if (closure.size()!=0) {
                        Classification_Rule rule = createClassificationRule(inst, instanceSubset, closure, attributes);
                        updateBestConcepts(bestConcepts, instanceSet, rule, closure);

                    }
                }
            }
        }

        // Construire la liste finale
        setConceptList(buildFinalConceptsList(inst, bestConcepts, allConcepts, rules)); 
           
        if (debug) {
            printConcepts(this.getConceptList(), inst);
        }
        return rules;
    }
    
    
    /**
     * Calcule la fermeture des attributs et retourne une Map<NomAttribut, Valeur>
     * Pour les datasets binaires, n'inclut que les attributs avec valeur 1
     */
    private Map<String, String> computeAttributeClosure(Instances inst, 
                                                      ArrayList<Integer> instances, 
                                                      List<Attribute> selectedAttributes) {
        Map<String, String> closureMap = new LinkedHashMap<>();

        if (instances.isEmpty()) {
            for (Attribute attr : selectedAttributes) {
                closureMap.put(attr.name(), "-null-");
            }
            return closureMap;
        }

        if (!isBinaryDataset(inst)) {
            // Cas général pour datasets non binaires
            for (Attribute attr : selectedAttributes) {
                String firstValue = inst.instance(instances.get(0)).stringValue(attr);
                boolean allSame = true;

                for (int j = 1; j < instances.size(); j++) {
                    if (!inst.instance(instances.get(j)).stringValue(attr).equals(firstValue)) {
                        allSame = false;
                        break;
                    }
                }

                closureMap.put(attr.name(), allSame ? firstValue : "-null-");
            }
        } else {
            // Cas spécial pour datasets binaires - on ignore les 0
            for (Attribute attr : selectedAttributes) {
                boolean allOne = true;

                for (int i : instances) {
                    if (inst.instance(i).value(attr) != 1.0) {
                        allOne = false;
                        break;
                    }
                }

                // On n'ajoute que les attributs avec valeur 1
                if (allOne) {
                    closureMap.put(attr.name(), "1");
                }
                // Les attributs avec 0 ne sont pas ajoutés du tout
            }
        }

        return closureMap;
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
                Map<String,String> closure = computeAttributeClosure(inst, instanceSubset,attributes);

                Classification_Rule rule = createClassificationRule(inst, instanceSubset, closure, attributes);
                updateBestConcepts(bestConcepts, instanceSet, rule, closure);
            }
        }

        // Construire la liste finale (même méthode que generateConceptsAllCombinations)
     // Construire la liste finale
        setConceptList(buildFinalConceptsList(inst, bestConcepts, allConcepts, rules)); 
        if (debug) {
            printConcepts(this.getConceptList(), inst);
        }
        
        return rules;
    }
    
    
    
    /*private List<List<Attribute>> generateAttributeCombinations(ArrayList<Attribute> attributes, int k) {
	    List<List<Attribute>> combinations = new ArrayList<>();
	    generateCombinationsHelper(attributes, k, 0, new ArrayList<>(), combinations);
	    return combinations;
	}*/
   
    
    private List<List<Attribute>> generateAttributeCombinations(Instances inst, ArrayList<Attribute> attributes, int k, boolean debug) {
        List<List<Attribute>> combinations = new ArrayList<>();

        // Si binaire, filtrer les attributs qui peuvent avoir 0, 0.0 ou false
        List<Attribute> filteredAttributes=attributes;
        if (isBinaryDataset(inst)) {
        	filteredAttributes = filterBinaryAttributes(attributes, inst);
        }

        generateCombinationsHelper(filteredAttributes, k, 0, new ArrayList<>(), combinations);

        if (debug) {
            System.out.println("=== ATTRIBUTE COMBINATIONS (k = " + k + ") ===");
            for (List<Attribute> combo : combinations) {
                String comboStr = combo.stream()
                    .map(Attribute::name)
                    .collect(Collectors.joining(", "));
                System.out.println(" - [" + comboStr + "]");
            }
            System.out.println("=== END COMBINATIONS ===");
        }

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
		 * Crée une règle de classification à partir d'une fermeture (ensemble d'attributs binaires),
		 * du sous-ensemble d'instances, et du dataset.
		 */
		private Classification_Rule createClassificationRule(Instances inst, 
		                                                   ArrayList<Integer> instanceSubset,
		                                                   Map<String,String> closure,
		                                                   ArrayList<Attribute> attributes) throws Exception {
		  
		    int majorityClassIndex = findMajorityClass(inst, instanceSubset);
		    
 		    Classification_Rule rule = new Classification_Rule(
		    		closure,
		        majorityClassIndex,
		        inst.classAttribute().value(majorityClassIndex)
		    );
		    

		    double weight = calculateRuleWeight(inst, rule);
		    rule.setRule_Ponderation(weight);

		    
		    
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
    
    private void updateBestConcepts(Map<Set<Integer>, Classification_Rule> bestConcepts,
            Set<Integer> instanceSet,
            Classification_Rule rule,
		    Map<String, String> closure) {
		if (bestConcepts.containsKey(instanceSet)) {
		Classification_Rule existingRule = bestConcepts.get(instanceSet);
		Map<String, String> existingClosure = existingRule.getAttributeValues();
		
		// Comparaison du nombre d'attributs spécifiques
		int newSpecific = countValidAttributes(closure);
		int existingSpecific = countValidAttributes(existingClosure);
		
		// Si nouvelle règle a plus d'attributs spécifiques OU
		// même nombre mais meilleure pondération
		if (newSpecific > existingSpecific || 
		(newSpecific == existingSpecific && 
		rule.getRule_Ponderation() > existingRule.getRule_Ponderation())) {
		bestConcepts.put(instanceSet, rule);
		}
		} else {
		bestConcepts.put(instanceSet, rule);
		}
		}
		    

    /**
     * Construit la liste finale des concepts en évitant les doublons de règles
     */
    private List<FormalConcept> buildFinalConceptsList(Instances inst,
                                     Map<Set<Integer>, Classification_Rule> bestConcepts,
                                     ArrayList<String> allConcepts,
                                     ArrayList<Classification_Rule> rules) {
        List<FormalConcept> concepts = new ArrayList<>();
        Set<String> existingRules = new HashSet<>(); // Pour suivre les règles déjà ajoutées
        
        for (Map.Entry<Set<Integer>, Classification_Rule> entry : bestConcepts.entrySet()) {
            Set<Integer> instanceSet = entry.getKey();
            if (instanceSet.size() <= 1) {
                continue;
            }
            
            Classification_Rule rule = entry.getValue();
            String ruleSignature = generateRuleSignature(rule); // Signature unique de la règle
            
            // Vérifie si la règle existe déjà
            if (!existingRules.contains(ruleSignature)) {
                existingRules.add(ruleSignature);
                
                FormalConcept concept = buildConcept(instanceSet, rule);
                String conceptStr = concept.toString();
                
                concepts.add(concept);
                allConcepts.add(conceptStr);
                rules.add(rule);
            }
        }
        
        // Export JSON
        String conceptsString = JsonExporter.toJson(concepts);
        String path = "C://Users//asus//Desktop//generated_concepts.json";
        JsonExporter.saveToFile(conceptsString, path);
        
        return concepts;
    }

    /**
     * Génère une signature unique pour une règle (comparaison de contenu)
     */
    private String generateRuleSignature(Classification_Rule rule) {
        StringBuilder signature = new StringBuilder();
        
        // Ajoute les prémisses triées
        List<String> premisses = new ArrayList<>(rule.getPremisses());
        Collections.sort(premisses);
        for (String prem : premisses) {
            signature.append(prem).append(";");
        }
        
        // Ajoute la conclusion
        signature.append("THEN:").append(rule.getRule_ClassMaj());
        
        return signature.toString();
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
     * Retourne les concepts groupés par niveau (nombre d'attributs) sous forme de chaîne.
     * @param allConcepts La liste complète des concepts
     * @return Chaîne représentant les concepts groupés par taille d'intention
     */
    public String getConceptLevelsAsString(List<FormalConcept> allConcepts) {
        Map<Integer, List<String>> conceptsByLevel = new TreeMap<>();
        StringBuilder sb = new StringBuilder();

        for (FormalConcept concept : allConcepts) {
            int level = concept.getAttributes().size();

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

        sb.append("\n=== CONCEPTS BY INTENTION SET SIZE ===\n");
        for (Map.Entry<Integer, List<String>> entry : conceptsByLevel.entrySet()) {
            sb.append(String.format("\n--- SIZE %d (%d concepts) ---\n",
                    entry.getKey(), entry.getValue().size()));

            int count = 0;
            for (String concept : entry.getValue()) {
                sb.append(concept).append("  ");
                if (++count % 5 == 0) sb.append("\n");
            }
            if (count % 5 != 0) sb.append("\n");
        }
        sb.append("=== END ===\n");

        return sb.toString();
    }
    public void afficheAttributsEtValeurs(Instances inst) {
        System.out.println("=== ATTRIBUTS ET LEURS VALEURS ===");

        for (int i = 0; i < inst.numAttributes(); i++) {
            Attribute attr = inst.attribute(i);
            System.out.print("- " + attr.name() + " : ");

            Set<String> valeurs = new TreeSet<>();

            for (int j = 0; j < inst.numInstances(); j++) {
                Instance instance = inst.instance(j);
                if (attr.isNumeric()) {
                    double val = instance.value(attr);
                    valeurs.add(String.valueOf(val));
                } else if (attr.isNominal()) {
                    valeurs.add(instance.stringValue(attr));
                }
            }

            System.out.println(valeurs);
        }

        System.out.println("=== FIN AFFICHAGE ===");
    }

    
    
  
    /**
     * Compte le nombre d'attributs valides dans une fermeture
     * @param closureMap Map des attributs (nom -> valeur)
     * @return Nombre d'attributs valides (non nuls)
     */
    private int countValidAttributes(Map<String, String> closureMap) {
        int count = 0;
        for (String value : closureMap.values()) {
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
            List<List<Attribute>> attrCombinations = generateAttributeCombinations(inst, attributes, k, debug);

            for (List<Attribute> attrCombination : attrCombinations) {
                ArrayList<Attribute> attrCombinationArrayList = new ArrayList<>(attrCombination);
                
                // Trouver les instances ayant les valeurs majoritaires pour ces attributs
                ArrayList<Integer> majorityInstances = findInstancesWithMajorityValues(inst, 
                    attrCombinationArrayList, majorityValues);

                if (majorityInstances.isEmpty()) continue;

                Set<Integer> instanceSet = new HashSet<>(majorityInstances);
                
                // Calculer la fermeture pour ces attributs
                Map<String,String> closure = computeAttributeClosure(inst, majorityInstances, attributes);

                // Créer la règle de classification
                Classification_Rule rule = createClassificationRule(inst, majorityInstances, closure, attributes);
                // Mettre à jour les meilleurs concepts
                updateBestConcepts(bestConcepts, instanceSet, rule, closure);
            }
        }

        // Construire la liste finale
        setConceptList(buildFinalConceptsList(inst, bestConcepts, allConcepts, rules)); 
           
        if (debug) {
            printConcepts(this.getConceptList(), inst);
        }
        return rules;
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
	                Map<String,String> closure = computeAttributeClosure(inst, matchingInstances,attributes);

	                // 5. Créer une règle spécifique à cet attribut
	                Classification_Rule rule = createClassificationRule(inst, matchingInstances, closure, attributes);
	                
	                // 6. Mettre à jour les meilleurs concepts
	                Set<Integer> instanceSet = new HashSet<>(matchingInstances);
	                updateBestConcepts(bestConcepts, instanceSet, rule, closure);
	            }
	        }

	        // 7. Construire le résultat final
	        setConceptList(buildFinalConceptsList(inst, bestConcepts, allConcepts, rules)); 
	           
	        if (debug) {
	            printConcepts(this.getConceptList(), inst);
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

        int size = rule.getPremisses().size();
        for (int i = 0; i < size; i++) {
            String attrValue = rule.getPremisses().get(i);

            if (!(attrValue.equals("-null-") || attrValue.equals("0") || attrValue.equals("0.0"))) {

                
                    labeledAttributes.add(attrValue);
                }
            
        }

          

        return new FormalConcept(instanceSet, labeledAttributes);
    }

    
}