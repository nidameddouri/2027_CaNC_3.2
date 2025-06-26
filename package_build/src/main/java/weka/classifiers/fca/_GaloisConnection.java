package weka.classifiers.fca;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Set;
import java.util.TreeMap;
import java.util.TreeSet;

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
       
        this.conceptList= new ArrayList<>();
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
    
  
    
    private Map<String, String> computeAttributeClosure(Instances inst, 
            ArrayList<Integer> instances, 
            List<Attribute> selectedAttributes) {
    	Map<String, String> closureMap = new LinkedHashMap<>();

		if (instances.isEmpty()) {
		return closureMap; // fermeture vide
		}

		if (!isBinaryDataset(inst)) {
		// Cas général pour datasets non binaires
		for (Attribute attr : selectedAttributes) {
		String firstValue = null;
		boolean allSame = true;
		
		for (int idx = 0; idx < instances.size(); idx++) {
		Instance instance = inst.instance(instances.get(idx));
		
		if (instance.isMissing(attr)) {
		allSame = false;
		break;
		}
		
		String val = instance.stringValue(attr);
		if (val == null || val.equals("?")) {
		allSame = false;
		break;
		}
		
		if (idx == 0) {
		firstValue = val;
		} else if (!val.equals(firstValue)) {
		allSame = false;
		break;
		}
		}
		
		if (allSame && firstValue != null) {
		closureMap.put(attr.name(), firstValue);
		}
		// Sinon : on ne met rien (pas de -null-)
		}
		} else {
		// Cas binaire : on inclut uniquement les attributs dont la valeur est 1 pour tous
		for (Attribute attr : selectedAttributes) {
		boolean allOne = true;
		
		for (int i : instances) {
		Instance instance = inst.instance(i);
		if (instance.isMissing(attr) || instance.value(attr) != 1.0) {
		allOne = false;
		break;
		}
		}
		
		if (allOne) {
		closureMap.put(attr.name(), "1");
		}
		// Sinon on ne met rien (valeurs 0 ou missing sont ignorées)
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
            //affichage concepts
        	printConcepts(this.getConceptList(), inst);
        }
        
        return rules;
    }
    
    
  
    
    private List<List<Attribute>> generateAttributeCombinations(Instances inst, ArrayList<Attribute> attributes, int k, boolean debug) {
        List<List<Attribute>> combinations = new ArrayList<>();

        // Si binaire, on filtre les attributs non discriminants
        List<Attribute> filteredAttributes = isBinaryDataset(inst)
            ? filterBinaryAttributes(attributes, inst)
            : attributes;

        int n = filteredAttributes.size();
        if (k > n) return combinations;

        if (debug) {
            System.out.println("➡ Generating attribute combinations of size " + k + " from " + n + " attributes");
        }

        // Initialiser un tableau d’indices [0, 1, ..., k-1]
        int[] indices = new int[k];
        for (int i = 0; i < k; i++) {
            indices[i] = i;
        }

        while (indices[0] <= n - k) {
            // Ajouter la combinaison courante
            List<Attribute> combination = new ArrayList<>();
            for (int index : indices) {
                combination.add(filteredAttributes.get(index));
            }
            combinations.add(combination);

            // Générer la prochaine combinaison
            int t = k - 1;
            while (t >= 0 && indices[t] == n - k + t) {
                t--;
            }
            if (t < 0) break;
            indices[t]++;
            for (int i = t + 1; i < k; i++) {
                indices[i] = indices[i - 1] + 1;
            }
        }

        return combinations;
    }



    private void generateCombinationsHelper(List<Attribute> attributes, int k, int start, 
            List<Attribute> current, List<List<Attribute>> result) {
    	if(m_Debug) {
    		System.out.println("Generating combination for "+k+" attributes");
    	}
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
        //String conceptsString = JsonExporter.toJson(concepts);
        //String conceptString= JsonExporter.toJsonGroupedByLevel(concepts);
        
        //String path = "//home//user//Stage//conceptsList.json";
        //JsonExporter.saveToFile(conceptString, path);
        
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
        System.out.println("=== ATTRIBUTES AND THEIR VALEURS ===");

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

        System.out.println("=== END ===");
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
     * Génère les concepts formels à partir des instances et attributs
     */
    ArrayList<Classification_Rule> generateConceptsAllCombinations(Instances inst, 
                                                         ArrayList<Attribute> attributes,
                                                         ArrayList<String> allConcepts,
                                                         boolean debug) throws Exception {
        Map<Set<Integer>, Classification_Rule> bestConcepts = new HashMap<>();
        ArrayList<Classification_Rule> rules = new ArrayList<>();
        
        //afficheAttributsEtValeurs(inst);
        if(debug) {
        	System.out.println("Generating concepts .....");
        }
        
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
            //affichage concepts
        	//printConcepts(this.getConceptList(), inst);
        }
        return rules;
    }

    /**
     * Génère les concepts formels à partir des combinaisons d'attributs avec leurs valeurs majoritaires
     */
    
    /**
     * Génère les concepts formels à partir des combinaisons d'attributs avec leurs valeurs majoritaires
     */
    public ArrayList<Classification_Rule> generateConceptsFromMajorityValuesAllCombinations(
            Instances inst, 
            ArrayList<Attribute> attributes,
            int Policy_Selection,
            ArrayList<String> allConcepts,
            boolean debug) throws Exception {

        if (debug) {
            System.out.println("🔍 Generating concepts from majority value combinations ...");
        }

        Map<Set<Integer>, Classification_Rule> bestConcepts = new HashMap<>();
        ArrayList<Classification_Rule> rules = new ArrayList<>();

        // 🔢 Étape 1 : Calcul des valeurs majoritaires pour chaque attribut
        if (debug) {
            System.out.println("🔢 Step 1: Calculating majority values for each attribute ...");
        }

        Map<Attribute, String> majorityValues = new HashMap<>();
        for (Attribute attr : attributes) {
            int majorityIndex = findMajorityValueIndex(inst, attr.index(), Policy_Selection);
            String majorityValue = attr.value(majorityIndex);
            majorityValues.put(attr, majorityValue);
            if (debug) {
                System.out.println("   - Majority value for attribute '" + attr.name() + "': " + majorityValue);
            }
        }

        // 🧩 Étape 2 : Générer les combinaisons d'attributs
        int m = attributes.size();
        for (int k = 1; k <= m; k++) {
            if (debug) {
                System.out.println("\n🧩 Step 2: Generating combinations of size " + k + " ...");
            }

            List<List<Attribute>> attrCombinations = generateAttributeCombinations(inst, attributes, k, debug);

            for (List<Attribute> attrCombination : attrCombinations) {
                ArrayList<Attribute> attrCombinationArrayList = new ArrayList<>(attrCombination);

                if (debug) {
                    System.out.print("   ➤ Processing combination: ");
                    for (Attribute attr : attrCombinationArrayList) {
                        System.out.print(attr.name() + "=" + majorityValues.get(attr) + "  ");
                    }
                    System.out.println();
                }

                // 🧮 Étape 3 : Trouver les instances correspondant à la combinaison
                ArrayList<Integer> instanceSubset = findInstancesWithMajorityValues(inst, attrCombinationArrayList, majorityValues);

                if (debug) {
                    System.out.println("      → Found " + instanceSubset.size() + " matching instances");
                }

                if (instanceSubset.size() <= 1) continue;

                Set<Integer> instanceSet = new HashSet<>(instanceSubset);
                Map<String, String> closure = computeAttributeClosure(inst, instanceSubset, attrCombinationArrayList);

                if (!closure.isEmpty()) {
                    Classification_Rule rule = createClassificationRule(inst, instanceSubset, closure, attributes);
                    updateBestConcepts(bestConcepts, instanceSet, rule, closure);
                }
            }
        }

        // 🧾 Étape 4 : Construire la liste finale
        setConceptList(buildFinalConceptsList(inst, bestConcepts, allConcepts, rules)); 

        if (debug) {
            System.out.println("\n📋 Final concepts:");
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
	        	//affichage concepts
	            //printConcepts(this.getConceptList(), inst);
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