package weka.classifiers.fca;

import java.text.SimpleDateFormat;

import java.util.ArrayList;
import java.util.Date;

import weka.attributeSelection.*;
import weka.core.Attribute;
import weka.core.Instances;
import weka.core.Utils;
import weka.core.converters.ConverterUtils.DataSource;


import java.util.stream.Collectors;

/**
 * Classe principale qui gère l'extraction des règles avec support pour les connexions de Galois
 */
public class RuleExtractor {
    private boolean m_Debug;
    private int FeatureMeasureMethod;
    private int Policy_Selection;
    private InfoGainAttributeEval m_InfoGainAttributeEval;
    private GainRatioAttributeEval m_GainRatioAttributeEval;
    // Constantes
    public static final int FEAT_ENTROPY = 1;
    public static final int FEAT_INFOGAIN = 2;
    public static final int FEAT_GAINRATIO = 3;
    
    public static final int POLICY_FIRST = 1;
    public static final int POLICY_LAST = 2;
    
    
    
    public static final int CONCEPT_LEARNING_ALL_ALL = 1;
    public static final int CONCEPT_LEARNING_ALL_MAJORITY = 2;
    public static final int CONCEPT_LEARNING_PERTINENT_ALL = 3;
    public static final int CONCEPT_LEARNING_PERTINENT_MAJORITY = 4;
    public static final int CONCEPT_LEARNING_ABOVE_THRESHOLD_ALL = 5;
    public static final int CONCEPT_LEARNING_ABOVE_THRESHOLD_MAJORITY = 6;
    public static final int CONCEPT_LEARNING_UNDER_THRESHOLD_ALL = 7;
    public static final int CONCEPT_LEARNING_UNDER_THRESHOLD_MAJORITY = 8;
    
  
    
    
    
    public RuleExtractor(boolean debug, int featureMeasureMethod, int policySelection, int conceptLearningStrategy) {
        this.m_Debug = debug;
        this.FeatureMeasureMethod = featureMeasureMethod;
        this.Policy_Selection = policySelection;
        this.m_InfoGainAttributeEval = new InfoGainAttributeEval();
        this.m_GainRatioAttributeEval = new GainRatioAttributeEval();
    
        if (debug) {
            String timestamp = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss").format(new Date());
            System.out.println("\n\n\n-----------[" + timestamp + "]-----------\n"
                + "RuleExtractor initialized with debug: " + m_Debug + ",\n"
                + "Concept Learning: " + conceptLearningToString(conceptLearningStrategy) + ",\n"
                + "Feature MeasureMethod: " + featureMeasureToString(featureMeasureMethod) + ",\n"
                + "Policy Selection: " + policySelectionToString(policySelection)
               );
        }
    }


    /**
     * Méthode pour convertir les stratégies de concept learning en chaînes
     */
    private String conceptLearningToString(int cl) {
        switch (cl) {
            case 1: return "Closure of All Attributes & All Values";
            case 2: return "Closure of ALL attributes & Majority Value";
            case 3: return "Closure of the Pertinent Attribute & All Values";
            case 4: return "Closure of the Pertinent attribute & Majority Value";
            case 5: return "Closure of attributes above THRESHOLD & All Values";
            case 6: return "Closure of attributes above THRESHOLD & Majority Value";
            case 7: return "Closure of attributes under THRESHOLD & All Values";
            case 8: return "Closure of attributes under THRESHOLD & Majority Value";
           
            default: return "Unknown";
        }
    }
    
    /**
     * Méthode pour convertir les mesures de caractéristiques en chaînes
     */
    private String featureMeasureToString(int fm) {
        switch (fm) {
            case 1: return "Entropy";
            case 2: return "Information Gain";
            case 3: return "Gain Ratio";
            default: return "Unknown";
        }
    }
    

    /**
     * Méthode pour convertir les politiques de sélection en chaînes
     */
    private String policySelectionToString(int p) {
        switch (p) {
            case 1: return "First attribute in case of equality";
            case 2: return "Last attribute in case of equality";
            default: return "Unknown";
        }
    }

    

    /**
     * Méthode pour afficher les fermetures au format ({o1, o2, o3}, {a1, a2})
     */
    public void printGaloisClosure(ArrayList<Integer> objects, ArrayList<String> attributes, Instances inst) {
        if (!m_Debug) return;

        System.out.println("\n--- COMPUTING GALOIS CONNEXION ---");
        
        // 1. Affichage des objets (instances)
        System.out.println("\nObjects (instances) :");
        for (Integer objIdx : objects) {
            System.out.print("  o" + (objIdx + 1) + " : ");
            for (int i = 0; i < inst.numAttributes(); i++) {
                if (i > 0) System.out.print(", ");
                System.out.print(inst.attribute(i).name() + "=" + 
                               inst.instance(objIdx).stringValue(i));
            }
            System.out.println();
        }
        
        // 2. Affichage des attributs de la fermeture
        System.out.println("\nConnexion Attributes: ");
        boolean hasAttributes = false;
        for (int i = 0; i < attributes.size(); i++) {
            if (!attributes.get(i).equals("-null-")) {
                System.out.println("  - " + inst.attribute(i).name() + " = " + attributes.get(i));
                hasAttributes = true;
            }
        }
        
        if (!hasAttributes) {
            System.out.println("  (No specific attributes)");
        }
        
        // 3. Résumé formel
        System.out.println("\nConcepts :");
        System.out.print("Connexion({o" + (objects.get(0) + 1));
        for (int i = 1; i < objects.size(); i++) {
            System.out.print(", o" + (objects.get(i) + 1));
        }
        System.out.print("}) = {");
        
        boolean firstAttr = true;
        for (int i = 0; i < attributes.size(); i++) {
            if (!attributes.get(i).equals("-null-")) {
                if (!firstAttr) System.out.print(", ");
                System.out.print(inst.attribute(i).name() + "=" + attributes.get(i));
                firstAttr = false;
            }
        }
        System.out.println("}");
    }
   


    
    /**
     * Initialise les évaluateurs d'attributs selon la méthode choisie
     */
    private void initializeFeatureEvaluator(Instances inst) throws Exception {
        switch (FeatureMeasureMethod) {
            case FEAT_ENTROPY:
                break;
            case FEAT_INFOGAIN:
                m_InfoGainAttributeEval.buildEvaluator(inst);
                break;
            case FEAT_GAINRATIO:
                m_GainRatioAttributeEval.buildEvaluator(inst);
                break;
        }
    }
    
   
    /**
     * Calcule les mesures pour chaque attribut
     */
    private double[] computeFeatureMeasures(Instances inst) throws Exception {
    double[] featureMeasures = new double[inst.numAttributes()];
    
    // Réinitialiser les évaluateurs à chaque appel
    m_InfoGainAttributeEval = new InfoGainAttributeEval();
    m_GainRatioAttributeEval = new GainRatioAttributeEval();
    
    // Initialisation obligatoire des évaluateurs
    m_InfoGainAttributeEval.buildEvaluator(inst);
    m_GainRatioAttributeEval.buildEvaluator(inst);

    // Calcul des mesures
    for (int i = 0; i < inst.numAttributes(); i++) {
        if (i == inst.classIndex()) continue;
        
        switch (FeatureMeasureMethod) {
            case FEAT_INFOGAIN:
                featureMeasures[i] = m_InfoGainAttributeEval.evaluateAttribute(i);
                break;
            case FEAT_GAINRATIO:
                featureMeasures[i] = m_GainRatioAttributeEval.evaluateAttribute(i);
                break;
            case FEAT_ENTROPY:
                featureMeasures[i] = computeEntropy(inst, inst.attribute(i));
                break;
            default:
                throw new IllegalStateException("Méthode de mesure inconnue");
        }
    }
    
    return featureMeasures;
}

    
    /**
     * Sélectionne l'attribut avec la meilleure mesure
     */
    private Attribute selectBestAttribute(Instances inst, double[] featureMeasures) {
        if (FeatureMeasureMethod == FEAT_ENTROPY) {
            int minIndex = Utils.minIndex(featureMeasures);
            if (minIndex == inst.classIndex()) {

                // Temporarily set class feature to max value so it's ignored
                double original = featureMeasures[minIndex];
                featureMeasures[minIndex] = Double.MAX_VALUE;
                minIndex = Utils.minIndex(featureMeasures);
                featureMeasures[inst.classIndex()] = original;
            }
            return inst.attribute(minIndex);
        } else {
            int maxIndex = Utils.maxIndex(featureMeasures);
            if (maxIndex == inst.classIndex()) {
                // Ignore class attribute
                double original = featureMeasures[maxIndex];
                featureMeasures[maxIndex] = Double.MIN_VALUE;
                maxIndex = Utils.maxIndex(featureMeasures);
                featureMeasures[inst.classIndex()] = original;
            }
            return inst.attribute(maxIndex);
        }
    }
    
    /**
     * Calcul d'entropie pour un attribut
     */
    private double computeEntropy(Instances inst, Attribute att) {
        double entropy = 0.0;
        int numClasses = inst.numClasses();
        int numValues = att.numValues();

        for (int i = 0; i < numValues; i++) {
            Instances subset = new Instances(inst, 0);
            for (int j = 0; j < inst.numInstances(); j++) {
                if (inst.instance(j).stringValue(att.index()).equals(att.value(i))) {
                    subset.add(inst.instance(j));
                }
            }

            if (subset.numInstances() > 0) {
                double[] classCounts = new double[numClasses];
                for (int j = 0; j < subset.numInstances(); j++) {
                    classCounts[(int) subset.instance(j).classValue()]++;
                }

                double subsetEntropy = 0.0;
                for (int j = 0; j < numClasses; j++) {
                    if (classCounts[j] > 0) {
                        double prob = classCounts[j] / subset.numInstances();
                        subsetEntropy -= prob * Utils.log2(prob);
                    }
                }

                entropy += ((double) subset.numInstances() / inst.numInstances()) * subsetEntropy;
            }
        }

        return entropy;
    }
    /**
     * Trouve l'index de la valeur majoritaire d'un attribut
     * @param inst Dataset complet
     * @param attrIdx Index de l'attribut à analyser
     * @return Index de la valeur majoritaire (selon Policy_Selection en cas d'égalité)
     */
    private int findMajorityValueIndex(Instances inst, int attrIdx) {
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
    
    
    private int countOccurrences(Instances inst, int attrIdx, int valueIdx) {
        int count = 0;
        for (int i = 0; i < inst.numInstances(); i++) {
            if ((int) inst.instance(i).value(attrIdx) == valueIdx) count++;
        }
        return count;
    }

    
    /**
     * Extrait des règles en utilisant tous les attributs mais seulement la valeur majoritaire all attributes Maj value
     */
    public ArrayList<Classification_Rule> extractAllAttributesMajorityValue(Instances inst) throws Exception {
        ArrayList<Classification_Rule> rules = new ArrayList<>();
        
        if (m_Debug) {
            System.out.println("\nExtracting rules with all attributes and majority values");
        }
        
        // Pour chaque attribut
        for (int attrIdx = 0; attrIdx < inst.numAttributes() - 1; attrIdx++) {
            Attribute attribute = inst.attribute(attrIdx);
            
            if (m_Debug) {
                System.out.println("\nProcessing attribute: " + attribute.name());
            }
            
            int majorityValueIdx = findMajorityValueIndex(inst, attrIdx);
            
            if (m_Debug) {
                System.out.println("Majority value: " + attribute.value(majorityValueIdx) + 
                                 " with count: " + countOccurrences(inst, attrIdx, majorityValueIdx));
            }
            
            // Calculer la fermeture pour la valeur majoritaire
            ArrayList<Integer> instances = new ArrayList<>();
            for (int i = 0; i < inst.numInstances(); i++) {
                if (inst.instance(i).value(attrIdx) == majorityValueIdx) {
                    instances.add(i);
                }
            }
            
            if (instances.isEmpty()) continue;
            
            // Calculer la fermeture des attributs
            ArrayList<String> attributes = computeAttributeClosure(inst, instances);
            
            // Afficher la fermeture de Galois
            printGaloisClosure(instances, attributes, inst);
            
            // Trouver la classe majoritaire
            int majorityClassIndex = findMajorityClass(inst, instances);
            
            if (m_Debug) {
                System.out.println("Majority class: " + inst.attribute(inst.classIndex()).value(majorityClassIndex));
            }
            
            // Créer la règle
            Classification_Rule rule = new Classification_Rule(
                attributes,
                majorityClassIndex,
                inst.attribute(inst.classIndex()).value(majorityClassIndex)
            );
            
            // Calculer la pondération
            double weight = calculateRuleWeight(inst, rule);
            rule.setRule_Ponderation(weight);
            
            // Afficher la régle crée

            if (m_Debug) {
                System.out.println("Rule: " + rule.affich_nom_rule(true));
            }
            
            rules.add(rule);
        }
        
        return rules;
    }
    
    /**
     * Extrait des règles en utilisant tous les attributs et toutes leurs valeurs all attributes all values
     */
    public ArrayList<Classification_Rule> extractAllAttributesAllValues(Instances inst) throws Exception {
        ArrayList<Classification_Rule> rules = new ArrayList<>();
        
        
        if (m_Debug) {
            System.out.println("\nExtracting rules with all attributes and all values");
        }
        
        // Pour chaque attribut
        for (int attrIdx = 0; attrIdx < inst.numAttributes() - 1; attrIdx++) {
            Attribute attribute = inst.attribute(attrIdx);
            
            if (m_Debug) {
                System.out.println("\nProcessing attribute: " + attribute.name());
            }
            
            // Pour chaque valeur de l'attribut
            for (int valueIdx = 0; valueIdx < attribute.numValues(); valueIdx++) {
                if (m_Debug) {
                    System.out.println("Processing value: " + attribute.value(valueIdx));
                }
                
                // Trouver les instances avec cette valeur
                ArrayList<Integer> instances = new ArrayList<>();
                for (int i = 0; i < inst.numInstances(); i++) {
                    if (inst.instance(i).value(attrIdx) == valueIdx) {
                        instances.add(i);
                    }
                }
                
                if (instances.isEmpty()) continue;
                
                // Calculer la fermeture des attributs
                ArrayList<String> attributes = computeAttributeClosure(inst, instances);
                
                // Afficher la fermeture de Galois
                printGaloisClosure(instances, attributes, inst);
                
                // Trouver la classe majoritaire
                int majorityClassIndex = findMajorityClass(inst, instances);
                
                if (m_Debug) {
                    System.out.println("Majority class: " + inst.attribute(inst.classIndex()).value(majorityClassIndex));
                }
              

                // Créer la règle
                Classification_Rule rule = new Classification_Rule(
                    attributes,
                    majorityClassIndex,
                    inst.attribute(inst.classIndex()).value(majorityClassIndex)
                );
               
                // Calculer la pondération
                double weight = calculateRuleWeight(inst, rule);
                rule.setRule_Ponderation(weight);
                // Afficher la régle crée

                if (m_Debug) {
                    System.out.println("Rule: " + rule.affich_nom_rule(true));
                }
                rules.add(rule);
            }
            if (m_Debug) {
            	System.out.println("══════════════════════════════════════════════════════════════════════════════════════════════════════");
            }
        }
        
        return rules;
    }
    
    /**
     * Méthode pour extraire des règles basées sur l attribut pertinant  et tout ses valeurs
     */
    public ArrayList<Classification_Rule> extractPertinentAttributeAllValues(Instances inst) throws Exception{
    	if(m_Debug) {
            //printDebugHeader(inst);
        }
    	// Initialiser les évaluateurs d'attributs selon la méthode choisie
        initializeFeatureEvaluator(inst);
        
        // Calculer les mesures pour chaque attribut
        double[] featureMeasures = computeFeatureMeasures(inst);
        Attribute bestAttribute = this.selectBestAttribute(inst, featureMeasures);
        if(m_Debug) {
            //printFeatureMeasures(inst, featureMeasures);
            //
        	printSelectedAttribute(inst, bestAttribute);
            
        }

        AllValuesClosureStrategy strategy = new AllValuesClosureStrategy(m_Debug, Policy_Selection, this);
        
        
        return strategy.extractRules(inst, bestAttribute, featureMeasures);

        

    }
    /**
     * Méthode pour extraire des règles basées sur l attribut pertinant  et  sa val maj
     */
    public ArrayList<Classification_Rule> extractPertinentAttributeMajorityValue(Instances inst) throws Exception{
    	//if(m_Debug) {
            //printDebugHeader(inst);
    	//}
    	// Initialiser les évaluateurs d'attributs selon la méthode choisie
        initializeFeatureEvaluator(inst);
        
        // Calculer les mesures pour chaque attribut
        double[] featureMeasures = computeFeatureMeasures(inst);
        Attribute bestAttribute = this.selectBestAttribute(inst, featureMeasures);

        
        //if(m_Debug) {
           // printFeatureMeasures(inst, featureMeasures);
        //}
        BestValueClosureStrategy strategy = new BestValueClosureStrategy(m_Debug, Policy_Selection, this);
        return strategy.extractRules(inst, bestAttribute, featureMeasures);


    }
    /**
     * Méthode pour extraire des règles basées sur les attributs depassant le seuil  et tous ses valeur
     */
	  public ArrayList<Classification_Rule> extractAttributeAboveThresholdAllValues(Instances inst, double threshold) throws Exception {
	        // utilisation de methode compute multiAtt closure
	        ArrayList<Classification_Rule> rules = new ArrayList<>();
	        double[] featureMeasures = computeFeatureMeasures(inst);
	        ArrayList<Attribute> topAttributes =  selectAttributesAboveThreshold(inst,  threshold);
	
	        // Nouvelle logique spécifique pour les attributs depassant le seuil
	        for (Attribute attr : topAttributes) {
	            for (int valueIdx = 0; valueIdx < attr.numValues(); valueIdx++) {
	                ArrayList<Integer> matchingInstances = getInstancesForValue(inst, attr, valueIdx);
	                
	                if (!matchingInstances.isEmpty()) {
	                    ArrayList<String> closure = computeMultiAttributeClosure(inst, matchingInstances, topAttributes);
	                    if (closure != null) {
	                        Classification_Rule rule = Classification_Rule.createRule(inst, matchingInstances, closure);
	                     // Afficher la régle crée

	                        if (m_Debug) {
	                            System.out.println("Rule: " + rule.affich_nom_rule(true));
	                        }
	                        rules.add(rule);
	                    }
	                }
	            }
	        }
	        
	        return rules;
	    }
    
    
	  /**
	     * Méthode pour extraire des règles basées sur les attributs depassant le seuil  et sa val maj
	     */
	  public ArrayList<Classification_Rule> extractAttributeAboveThresholdMajorityValues(Instances inst, double threshold) throws Exception {
		    ArrayList<Classification_Rule> rules = new ArrayList<>();
		    double[] featureMeasures = computeFeatureMeasures(inst);
		    ArrayList<Attribute> topAttributes = selectAttributesAboveThreshold(inst, threshold);

		    if (m_Debug) {
		        System.out.println("\n=== EXTRACTING RULES (Threshold=" + threshold + ") ===");
		        System.out.println("Selected attributes: " + topAttributes.stream()
		                           .map(Attribute::name)
		                           .collect(Collectors.joining(", ")));
		    }

		    for (Attribute attr : topAttributes) {
		        // Trouver la valeur majoritaire pour l'attribut courant
		        int majorityValueIdx = findMajorityValueIndex(inst, attr.index());
		        String majorityValue = attr.value(majorityValueIdx);

		        if (m_Debug) {
		            System.out.println("\nProcessing attribute: " + attr.name() + 
		                             " (Majority value: " + majorityValue + ")");
		        }

		        // Récupérer les instances avec la valeur majoritaire
		        ArrayList<Integer> matchingInstances = new ArrayList<>();
		        for (int i = 0; i < inst.numInstances(); i++) {
		            if ((int)inst.instance(i).value(attr) == majorityValueIdx) {
		                matchingInstances.add(i);
		            }
		        }

		        if (!matchingInstances.isEmpty()) {
		            // Calculer la fermeture de Galois sur ces instances
		            ArrayList<String> closure = computeAttributeClosure(inst, matchingInstances);

		            if (m_Debug) {
		                System.out.println("Galois closure for " + attr.name() + "=" + majorityValue + ":");
		                printGaloisClosure(matchingInstances, closure, inst);
		            }

		            // Trouver la classe majoritaire parmi ces instances
		            int majorityClassIndex = findMajorityClass(inst, matchingInstances);
		            String className = inst.attribute(inst.classIndex()).value(majorityClassIndex);

		            // Créer la règle
		            Classification_Rule rule = new Classification_Rule(
		                closure,
		                majorityClassIndex,
		                className
		            );
		            
		            // Calculer la pondération
		            double weight = calculateRuleWeight(inst, rule);
		            rule.setRule_Ponderation(weight);

		            if (m_Debug) {
		                System.out.println("Created rule: " + rule.affich_nom_rule(true));
		            }

		            rules.add(rule);
		        }
		    }

		    if (m_Debug) {
		        System.out.println("\nTotal rules extracted: " + rules.size());
		    }
		    
		    return rules;
		}
    
	  /**
	     * Méthode pour extraire des règles basées sur les attributs depassant le seuil  et tous ses valeur
	     */
		public ArrayList<Classification_Rule> extractAttributeUnderThresholdAllValues(Instances inst, double threshold) throws Exception {
			// utilisation de methode compute multiAtt closure
	        ArrayList<Classification_Rule> rules = new ArrayList<>();
	        double[] featureMeasures = computeFeatureMeasures(inst);
	        ArrayList<Attribute> topAttributes =  selectAttributesUnderThreshold(inst,  threshold);
	        if (m_Debug) {
		        System.out.println("\n=== EXTRACTING RULES (Under Threshold=" + threshold + ") ===");
		        System.out.println("Selected attributes: " + topAttributes.stream()
		                           .map(Attribute::name)
		                           .collect(Collectors.joining(", ")));
		    }

	        // Nouvelle logique spécifique pour les attributs depassant le seuil
	        for (Attribute attr : topAttributes) {
	        	if (m_Debug) {
		            System.out.println("\nProcessing attribute: " + attr.name() );
		        }
	            for (int valueIdx = 0; valueIdx < attr.numValues(); valueIdx++) {
	                ArrayList<Integer> matchingInstances = getInstancesForValue(inst, attr, valueIdx);
	                
	                if (!matchingInstances.isEmpty()) {
	                    ArrayList<String> closure = computeMultiAttributeClosure(inst, matchingInstances, topAttributes);
	                    if (m_Debug) {
			                System.out.println("Galois closure for " + attr.name() + ":");
			                printGaloisClosure(matchingInstances, closure, inst);
			            }
	                    if (closure != null) {
	                        Classification_Rule rule = Classification_Rule.createRule(inst, matchingInstances, closure);
	                     // Afficher la régle crée

	                        if (m_Debug) {
	                            System.out.println("Rule: " + rule.affich_nom_rule(true));
	                        }
	                        rules.add(rule);
	                    }
	                }
	            }
	        }
	        if (m_Debug) {
		        System.out.println("\nTotal rules extracted: " + rules.size());
		    }
	        return rules;
	    }
	  /**
	     * Méthode pour extraire des règles basées sur les attributs depassant le seuil  et tous ses valeur
	     */
			public ArrayList<Classification_Rule> extractAttributeUnderThresholdMajorityValues(Instances inst, double threshold) throws Exception {
				ArrayList<Classification_Rule> rules = new ArrayList<>();
			    ArrayList<Attribute> topAttributes = selectAttributesUnderThreshold(inst, threshold);

			    if (m_Debug) {
			        System.out.println("\n=== EXTRACTING RULES (Under Threshold=" + threshold + ") ===");
			        System.out.println("Selected attributes: " + topAttributes.stream()
			                           .map(Attribute::name)
			                           .collect(Collectors.joining(", ")));
			    }

			    for (Attribute attr : topAttributes) {
			        // Trouver la valeur majoritaire pour l'attribut courant
			        int majorityValueIdx = findMajorityValueIndex(inst, attr.index());
			        String majorityValue = attr.value(majorityValueIdx);

			        if (m_Debug) {
			            System.out.println("\nProcessing attribute: " + attr.name() + 
			                             " (Majority value: " + majorityValue + ")");
			        }

			        // Récupérer les instances avec la valeur majoritaire
			        ArrayList<Integer> matchingInstances = new ArrayList<>();
			        for (int i = 0; i < inst.numInstances(); i++) {
			            if ((int)inst.instance(i).value(attr) == majorityValueIdx) {
			                matchingInstances.add(i);
			            }
			        }

			        if (!matchingInstances.isEmpty()) {
			            // Calculer la fermeture de Galois sur ces instances
			            ArrayList<String> closure = computeAttributeClosure(inst, matchingInstances);

			            if (m_Debug) {
			                System.out.println("Galois closure for " + attr.name() + "=" + majorityValue + ":");
			                printGaloisClosure(matchingInstances, closure, inst);
			            }

			            // Trouver la classe majoritaire parmi ces instances
			            int majorityClassIndex = findMajorityClass(inst, matchingInstances);
			            String className = inst.attribute(inst.classIndex()).value(majorityClassIndex);

			            // Créer la règle
			            Classification_Rule rule = new Classification_Rule(
			                closure,
			                majorityClassIndex,
			                className
			            );
			            
			            // Calculer la pondération
			            double weight = calculateRuleWeight(inst, rule);
			            rule.setRule_Ponderation(weight);

			            if (m_Debug) {
			                System.out.println("Created rule: " + rule.affich_nom_rule(true));
			            }

			            rules.add(rule);
			        }
			    }

			    if (m_Debug) {
			        System.out.println("\nTotal rules extracted: " + rules.size());
			    }
			    
			    return rules;
			}
  
 
    /* a supprimer*/
    public ArrayList<Classification_Rule> ExtraireRegleFermNom(Instances inst, int critere) throws Exception {        
        if(m_Debug) {
            printDebugHeader(inst);
        }
        
        // Initialiser les évaluateurs d'attributs selon la méthode choisie
        initializeFeatureEvaluator(inst);
        
        // Calculer les mesures pour chaque attribut
        double[] featureMeasures = computeFeatureMeasures(inst);
        
        if(m_Debug) {
            printFeatureMeasures(inst, featureMeasures);
        }
        
        // Sélectionner l'attribut avec la meilleure mesure
        Attribute bestAttribute = selectBestAttribute(inst, featureMeasures);
        
        if(m_Debug) {
            printSelectedAttribute(inst, bestAttribute);
        }
        
        // Appliquer la stratégie d'extraction de règles selon le critère spécifié
        RuleExtractionStrategy strategy;
        switch(critere) {
            case 1:
                strategy = new BestValueClosureStrategy(m_Debug, Policy_Selection, this);
                break;
            case 2:
                strategy = new ClassBasedClosureStrategy(m_Debug, Policy_Selection, this);
                break;
            case 3:
                strategy = new AllValuesClosureStrategy(m_Debug, Policy_Selection, this);
                break;
            default:
                throw new IllegalArgumentException("Critère d'extraction de règles non reconnu: " + critere);
        }
        
        return strategy.extractRules(inst, bestAttribute, featureMeasures);
    }

    
    
    
    /**
     * Extrait des règles en considérant tous les attributs par classe
     */
    public ArrayList<Classification_Rule> extractAllAttributesClassBased(Instances inst) throws Exception {
        ArrayList<Classification_Rule> rules = new ArrayList<>();
        
        if (m_Debug) {
            System.out.println("\nExtracting rules with all attributes based on class information");
        }
        
        // Pour chaque classe
        for (int classIdx = 0; classIdx < inst.numClasses(); classIdx++) {
            if (m_Debug) {
                System.out.println("\nProcessing class: " + inst.attribute(inst.classIndex()).value(classIdx));
            }
            
            // Collecter les instances de cette classe
            ArrayList<Integer> classInstances = new ArrayList<>();
            for (int i = 0; i < inst.numInstances(); i++) {
                if (inst.instance(i).classValue() == classIdx) {
                    classInstances.add(i);
                }
            }
            
            if (classInstances.isEmpty()) continue;
            
            // Pour chaque attribut
            for (int attrIdx = 0; attrIdx < inst.numAttributes() - 1; attrIdx++) {
                Attribute attribute = inst.attribute(attrIdx);
                
                if (m_Debug) {
                    System.out.println("Processing attribute: " + attribute.name());
                }
                
                // Trouver la valeur la plus fréquente pour cet attribut dans cette classe
                int[] valueCounts = new int[attribute.numValues()];
                for (int i : classInstances) {
                    int valueIdx = (int) inst.instance(i).value(attrIdx);
                    valueCounts[valueIdx]++;
                }
                
                int majorityValueIdx = 0;
                for (int i = 0; i < valueCounts.length; i++) {
                    if (valueCounts[i] > valueCounts[majorityValueIdx]) {
                        majorityValueIdx = i;
                    } else if (valueCounts[i] == valueCounts[majorityValueIdx] && Policy_Selection == POLICY_LAST) {
                        majorityValueIdx = i;
                    }
                }
                
                // Collecter les instances de cette classe avec cette valeur d'attribut
                ArrayList<Integer> relevantInstances = new ArrayList<>();
                for (int i : classInstances) {
                    if (inst.instance(i).value(attrIdx) == majorityValueIdx) {
                        relevantInstances.add(i);
                    }
                }
                
                if (relevantInstances.isEmpty()) continue;
                
                // Calculer la fermeture des attributs
                ArrayList<String> attributes = computeAttributeClosure(inst, relevantInstances);
                
                // Afficher la fermeture de Galois
                printGaloisClosure(relevantInstances, attributes, inst);
                
                // Créer la règle
                Classification_Rule rule = new Classification_Rule(
                    attributes,
                    classIdx,
                    inst.attribute(inst.classIndex()).value(classIdx)
                );
                
                // Calculer la pondération
                double weight = calculateRuleWeight(inst, rule);
                rule.setRule_Ponderation(weight);
                
                rules.add(rule);
            }
        }
        
        return rules;
    }
    
    /**
     * Extrait des règles en utilisant une connexion de Galois complète 
     */
    public ArrayList<Classification_Rule> extractCompleteGalois(Instances inst) throws Exception {
        ArrayList<Classification_Rule> rules = new ArrayList<>();
        
        if (m_Debug) {
            System.out.println("\nExtracting rules with complete Galois connection");
        }
        
        // Pour chaque classe
        for (int classIdx = 0; classIdx < inst.numClasses(); classIdx++) {
            if (m_Debug) {
                System.out.println("\nProcessing class: " + inst.attribute(inst.classIndex()).value(classIdx));
            }
            
            // Pour chaque attribut
            for (int attrIdx = 0; attrIdx < inst.numAttributes() - 1; attrIdx++) {
                Attribute attribute = inst.attribute(attrIdx);
                
                if (m_Debug) {
                    System.out.println("Processing attribute: " + attribute.name());
                }
                
                // Pour chaque valeur de l'attribut
                for (int valueIdx = 0; valueIdx < attribute.numValues(); valueIdx++) {
                    if (m_Debug) {
                        System.out.println("Processing value: " + attribute.value(valueIdx));
                    }
                    
                    // Collecter les instances de cette classe avec cette valeur d'attribut
                    ArrayList<Integer> relevantInstances = new ArrayList<>();
                    for (int i = 0; i < inst.numInstances(); i++) {
                        if (inst.instance(i).classValue() == classIdx && 
                            inst.instance(i).value(attrIdx) == valueIdx) {
                            relevantInstances.add(i);
                        }
                    }
                    
                    if (relevantInstances.isEmpty()) continue;
                    
                    // Calculer la fermeture des attributs
                    ArrayList<String> attributes = computeAttributeClosure(inst, relevantInstances);
                    
                    // Afficher la fermeture de Galois
                    printGaloisClosure(relevantInstances, attributes, inst);
                    
                    // Créer la règle
                    Classification_Rule rule = new Classification_Rule(
                        attributes,
                        classIdx,
                        inst.attribute(inst.classIndex()).value(classIdx)
                    );
                    
                    // Calculer la pondération
                    double weight = calculateRuleWeight(inst, rule);
                    rule.setRule_Ponderation(weight);
                    
                    rules.add(rule);
                }
            }
        }
        
        return rules;
    }
    
    
    /**
     * Génère tous les concepts formels pour les attributs sélectionnés
     */
    public ArrayList<FormalConcept> generateFormalConcepts(Instances inst, 
                                                         ArrayList<Attribute> selectedAttributes) {
        ArrayList<FormalConcept> concepts = new ArrayList<>();
        
        // 1. Générer toutes les combinaisons non vides d'attributs
        ArrayList<ArrayList<Attribute>> attributeCombinations = generateAttributeCombinations(selectedAttributes);
        
        // 2. Pour chaque combinaison d'attributs
        for (ArrayList<Attribute> attrCombination : attributeCombinations) {
            // 3. Trouver toutes les instances qui partagent ces attributs
            ArrayList<Integer> commonInstances = findCommonInstances(inst, attrCombination);
            
            if (!commonInstances.isEmpty()) {
                // 4. Calculer la fermeture complète
                ArrayList<String> fullClosure = computeFullClosure(inst, commonInstances);
                
                // 5. Créer et stocker le concept
                concepts.add(new FormalConcept(commonInstances, fullClosure));
            }
        }
        
        return concepts;
    }

    /**
     * Génère toutes les combinaisons non vides d'attributs (puissance ensemble - ∅)
     */
    private ArrayList<ArrayList<Attribute>> generateAttributeCombinations(ArrayList<Attribute> attributes) {
        ArrayList<ArrayList<Attribute>> combinations = new ArrayList<>();
        int n = attributes.size();
        
        // Générer toutes les combinaisons possibles (2^n - 1)
        for (int i = 1; i < (1 << n); i++) {
            ArrayList<Attribute> current = new ArrayList<>();
            for (int j = 0; j < n; j++) {
                if ((i & (1 << j)) > 0) {
                    current.add(attributes.get(j));
                }
            }
            combinations.add(current);
        }
        
        return combinations;
    }

    /**
     * Trouve les instances partageant tous les attributs de la combinaison
     */
    private ArrayList<Integer> findCommonInstances(Instances inst, ArrayList<Attribute> attributes) {
        ArrayList<Integer> commonInstances = new ArrayList<>();
        
        // Pour chaque instance
        instanceLoop:
        for (int i = 0; i < inst.numInstances(); i++) {
            // Vérifier si elle possède tous les attributs de la combinaison
            for (Attribute attr : attributes) {
                String value = inst.instance(i).stringValue(attr);
                boolean hasAttribute = false;
                
                // Vérifier si la valeur fait partie des valeurs possibles
                for (int j = 0; j < attr.numValues(); j++) {
                    if (attr.value(j).equals(value)) {
                        hasAttribute = true;
                        break;
                    }
                }
                
                if (!hasAttribute) continue instanceLoop;
            }
            commonInstances.add(i);
        }
        
        return commonInstances;
    }

    /**
     * Calcule la fermeture complète pour les instances données
     */
    private ArrayList<String> computeFullClosure(Instances inst, ArrayList<Integer> instances) {
        ArrayList<String> closure = new ArrayList<>();
        
        // Initialiser avec -null- pour tous les attributs
        for (int i = 0; i < inst.numAttributes(); i++) {
            if (i != inst.classIndex()) {
                closure.add("-null-");
            }
        }
        
        // Pour chaque attribut (sauf classe)
        for (int i = 0; i < inst.numAttributes(); i++) {
            if (i == inst.classIndex()) continue;
            
            Attribute attr = inst.attribute(i);
            String firstValue = inst.instance(instances.get(0)).stringValue(attr);
            boolean isConstant = true;
            
            for (int instIdx : instances) {
                if (!inst.instance(instIdx).stringValue(attr).equals(firstValue)) {
                    isConstant = false;
                    break;
                }
            }
            
            if (isConstant) {
                int closureIdx = (i < inst.classIndex()) ? i : i - 1;
                closure.set(closureIdx, firstValue);
            }
        }
        
        return closure;
    }
   
    /**
     * Affiche tous les concepts formels
     */
    public void displayFormalConcepts(Instances inst, ArrayList<FormalConcept> concepts) {
        System.out.println("\n=== FORMALS CONCEPTS  ===");
        System.out.println("Number of concepts: " + concepts.size());
        
        for (FormalConcept concept : concepts) {
            System.out.println(concept.toString(inst));
            System.out.println("----------------------");
        }
    }
    
    
    /**
     * Calcule la fermeture des attributs pour un ensemble d'instances
     */
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


    public static String repeat(String str, int times) {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < times; i++) sb.append(str);
        return sb.toString();
    }
    
    public static void printDatasetTable(Instances inst) {
        int colPadding = 3;
        int[] colWidths = new int[inst.numAttributes()];
        int instanceColWidth = String.valueOf(inst.numInstances()).length();
    
        for (int j = 0; j < inst.numAttributes(); j++) {
            colWidths[j] = inst.attribute(j).name().length();
        }
    
        for (int i = 0; i < inst.numInstances(); i++) {
            for (int j = 0; j < inst.numAttributes(); j++) {
                int len = inst.instance(i).stringValue(j).length();
                if (len > colWidths[j]) {
                    colWidths[j] = len;
                }
            }
        }
    
        // En-tete
        System.out.printf("%-" + (instanceColWidth + colPadding) + "s", "Inst");
        for (int j = 0; j < inst.numAttributes(); j++) {
            System.out.printf("%-" + (colWidths[j] + colPadding) + "s", inst.attribute(j).name());
        }
        System.out.println();
    
        
        int totalWidth = instanceColWidth + colPadding;
        for (int w : colWidths) totalWidth += w + colPadding;
        System.out.println(repeat("-", totalWidth));
    
        // Données
        for (int i = 0; i < inst.numInstances(); i++) {
            System.out.printf("%-" + (instanceColWidth + colPadding) + "d", (i + 1));
            for (int j = 0; j < inst.numAttributes(); j++) {
                System.out.printf("%-" + (colWidths[j] + colPadding) + "s", inst.instance(i).stringValue(j));
            }
            System.out.println();
        }
    
        // Ligne de fin
        System.out.println(repeat("-", totalWidth));
    }
    
    
    // Méthodes d'affichage pour le debug
    // On affiche un joli ASCII art de coq au début pour s'y retrouver dans les logs (merci Nida pour l'idee du coq, c'était un canard à la base)
    private void printDebugHeader(Instances inst) {
        System.out.println("      ,~.\n" +
                           "   ,-'__ `-,\n" +
                           "  {,-'  `. }              ,')\n" +
                           " ,( a )   `-.__         ,','),~,\n" +
                           "<=.) (         `-.__,==' ' ' '}\n" +
                           "  (   )                      /\n" +
                           "   `-\\   ,                  )\n" +
                           "       |  \\        `~.      /\n" +
                           "       \\   `._        \\    /\n" +
                           "        \\     `._____,'   /\n" +
                           "         `-.            ,'\n" +
                           "            `-.      ,-'\n" +
                           "               `~~~~'\n" +
                           "               //_||\n" +
                           "            __//--'/`\n" +
                           "          ,--'/`  '\n" +
                           "             '\n");
        System.out.println("\n--- Computing Galois Connexion ---\n");
       // System.out.println("Liste des attributs d'entrée:\n");
        //printDatasetTable(inst);
        //System.out.println("\n\n");
    }
    
    private void printFeatureMeasures(Instances inst, double[] featureMeasures) {
        String featMeasureName = "";
        switch (FeatureMeasureMethod) {
            case FEAT_ENTROPY: featMeasureName = "Entropies"; break;
            case FEAT_INFOGAIN: featMeasureName = "InfoGains"; break;
            case FEAT_GAINRATIO: featMeasureName = "GainRatios"; break;
        }

        System.out.println("\n--- CALCUL DE LA MESURE D'EXTRACTION ---\n");

        System.out.println("Mesure choisie pour l'extraction des regles: " + featMeasureName);
        System.out.println("Mesures de chaque attribut de ce contexte:");

        for(int i=0; i<inst.numAttributes(); i++) {
            if (i != inst.classIndex()) {
                System.out.println("\t" + inst.attribute(i).name() + " : " + featureMeasures[i]);
            }
        }
    }
    
    private void printSelectedAttribute(Instances inst, Attribute attribute) {
        System.out.println("\nL'attribut retenu pour calculer la fermeture est : " + attribute.name() + 
                           " (index: " + attribute.index() + ")");

        System.out.println("\tCet attribut peut avoir " + inst.numDistinctValues(attribute.index()) + 
                           " valeurs possibles.");
        for(int i=0; i<inst.numDistinctValues(attribute.index()); i++)
            System.out.println("\t\tValeur "+(i+1)+" : "+inst.attribute(attribute.index()).value(i));

        System.out.println("\n");
    }

    
    
    /**
     * Selectionne les attributs qui dépassent le seuil donné en paramètres
     * 
     * @param inst Les instances du dataset
     * @param measure measure calculés pour tous les attributs
     * @param threshold seuil
     */
    
    private ArrayList<Attribute> selectAttributesAboveThreshold(Instances inst, double threshold) {
    ArrayList<Attribute> selectedAttributes = new ArrayList<>();
    double[] measures= null;
	try {
		measures = computeFeatureMeasures(inst);
	} catch (Exception e) {
		System.out.println("Error in measures calculating in selectAttributesAboveThreshold function: "+e.getMessage());
	}

    for (int i = 0; i < measures.length; i++) {
        if (i != inst.classIndex()) {
            double score = measures[i];

            // Vérifie si le score dépasse le seuil 
            if ( score >= threshold) {
                selectedAttributes.add(inst.attribute(i));
            }
        }
    }

    return selectedAttributes;
}

    private ArrayList<Attribute> selectAttributesUnderThreshold(Instances inst, double threshold) {
    ArrayList<Attribute> selectedAttributes = new ArrayList<>();
    double[] measures= null;
	try {
		measures = computeFeatureMeasures(inst);
	} catch (Exception e) {
		System.out.println("Error in measures calculating in selectAttributesAboveThreshold function: "+e.getMessage());
	}

    for (int i = 0; i < measures.length; i++) {
        if (i != inst.classIndex()) {
            double score = measures[i];

         // Vérifie si le score dépasse le seuil 
            if ( score <= threshold) {
                selectedAttributes.add(inst.attribute(i));
            }
        }
    }

    return selectedAttributes;
}

    
    
    public void printAttributesUnderThreshold(Instances inst, double threshold) throws Exception {
        // Calculer les mesures pour chaque attribut
        double[] featureMeasures = computeFeatureMeasures(inst);

        // Créer une liste de tous les attributs avec leurs mesures (sauf la classe)
        ArrayList<Pair<Attribute, Double>> allAttributeMeasures = new ArrayList<>();
        for (int i = 0; i < inst.numAttributes(); i++) {
            if (i != inst.classIndex()) {
                allAttributeMeasures.add(new Pair<>(inst.attribute(i), featureMeasures[i]));
            }
        }

        // Trier tous les attributs selon la métrique
        allAttributeMeasures.sort((a1, a2) -> {
            if (FeatureMeasureMethod == FEAT_ENTROPY) {
                return Double.compare(a1.getValue(), a2.getValue());
            } else {
                return Double.compare(a2.getValue(), a1.getValue());
            }
        });

        // Affichage de TOUS les attributs
        System.out.println("\n▬▬▬▬▬▬▬▬▬▬▬▬▬ ALL ATTRIBUTES ▬▬▬▬▬▬▬▬▬▬▬▬▬");
        System.out.println("[" + featureMeasureToString(FeatureMeasureMethod) + "]\n");
        System.out.println(String.format("%-6s%-30s%12s", "Rank", "Attribute", "Value"));
        System.out.println("━".repeat(48));
        
        int rank = 1;
        for (Pair<Attribute, Double> p : allAttributeMeasures) {
            String cleanAttrName = p.getKey().name().trim().replaceAll("\\s+", " ");
            System.out.println(String.format("%-6d%-30s%12.4f",
                rank++,
                cleanAttrName.length() > 30 ? cleanAttrName.substring(0, 27) + "..." : cleanAttrName,
                p.getValue()));
        }
        
        System.out.println("-------------------------------------------");

        
        // Statistiques de tous les attributs
        double max = allAttributeMeasures.get(0).getValue();
        double min = allAttributeMeasures.get(allAttributeMeasures.size() - 1).getValue();
        double average = allAttributeMeasures.stream()
            .mapToDouble(Pair::getValue)
            .average()
            .orElse(0);

        System.out.println("\n📊 Statistics:");
        System.out.println(String.format("\nMax: %8.4f  Min: %8.4f  Moy: %8.4f", max, min, average));

        // Obtenir les attributs sous le seuil
        ArrayList<Attribute> selectedAttributes = new ArrayList<>();
        for (Pair<Attribute, Double> p : allAttributeMeasures) {
            if (p.getValue() < threshold) {
                selectedAttributes.add(p.getKey());
            }
        }

        if (selectedAttributes.isEmpty()) {
            System.out.println("\n⚠ No attributes below threshold: " + threshold);
            return;
        }

        // Affichage des attributs sous le seuil
        System.out.println("\n▬▬▬▬▬▬▬▬ ATTRIBUTES UNDER THRESHOLD (" + threshold + ") ▬▬▬▬▬▬▬▬");
        System.out.println("[" + featureMeasureToString(FeatureMeasureMethod) + "]\n");
        System.out.println("Selected attributes: " + selectedAttributes.size() + "/" + allAttributeMeasures.size() + "\n");

        System.out.println(String.format("%-6s%-30s%12s", "Rank", "Attribute", "Value"));
        System.out.println("━".repeat(48));

        int displayedRank = 1;
        for (Pair<Attribute, Double> p : allAttributeMeasures) {
            if (p.getValue() < threshold) {
                String cleanAttrName = p.getKey().name().trim().replaceAll("\\s+", " ");
                System.out.println(String.format("%-6d%-30s%12.4f",
                    displayedRank++,
                    cleanAttrName.length() > 30 ? cleanAttrName.substring(0, 27) + "..." : cleanAttrName,
                    p.getValue()));
            }
        }
    }
    
    public void printPertinentAttribute(Instances inst) throws Exception {
       
    	// Calculer les mesures pour chaque attribut
        double[] featureMeasures = computeFeatureMeasures(inst);

        // Créer une liste de tous les attributs avec leurs mesures (sauf la classe)
        ArrayList<Pair<Attribute, Double>> allAttributeMeasures = new ArrayList<>();
        for (int i = 0; i < inst.numAttributes(); i++) {
            if (i != inst.classIndex()) {
                allAttributeMeasures.add(new Pair<>(inst.attribute(i), featureMeasures[i]));
            }
        }

        // Trier tous les attributs selon la métrique
        allAttributeMeasures.sort((a1, a2) -> {
            if (FeatureMeasureMethod == FEAT_ENTROPY) {
                return Double.compare(a1.getValue(), a2.getValue());
            } else {
                return Double.compare(a2.getValue(), a1.getValue());
            }
        });

        // Affichage de TOUS les attributs
        System.out.println("\n▬▬▬▬▬▬▬▬▬▬▬▬▬ ALL ATTRIBUTES ▬▬▬▬▬▬▬▬▬▬▬▬▬");
        System.out.println("[" + featureMeasureToString(FeatureMeasureMethod) + "]\n");
        System.out.println(String.format("%-6s%-30s%12s", "Rank", "Attribute", "Value"));
        System.out.println("━".repeat(48));
        
        int rank = 1;
        for (Pair<Attribute, Double> p : allAttributeMeasures) {
            String cleanAttrName = p.getKey().name().trim().replaceAll("\\s+", " ");
            System.out.println(String.format("%-6d%-30s%12.4f",
                rank++,
                cleanAttrName.length() > 30 ? cleanAttrName.substring(0, 27) + "..." : cleanAttrName,
                p.getValue()));
        }

            System.out.println("\n=== MOST PERTINENT ATTRIBUTE ===");
            
            if (allAttributeMeasures == null || allAttributeMeasures.isEmpty()) {
                System.out.println("No attributes available!");
                return;
            }

            // Sélection basée sur la métrique
            Pair<Attribute, Double> pertinentPair;
            String criteria;
            
            if (FeatureMeasureMethod == FEAT_ENTROPY) {
                pertinentPair = allAttributeMeasures.get(allAttributeMeasures.size() - 1);
                criteria = "Lowest Entropy";
            } else {
                pertinentPair = allAttributeMeasures.get(0);
                criteria = "Highest " + featureMeasureToString(FeatureMeasureMethod);
            }

            // Affichage formaté
            System.out.println("Selection Criteria: " + criteria);
            System.out.println("Attribute: " + pertinentPair.getKey().name().trim());
            System.out.printf("Measure Value: %.4f%n", pertinentPair.getValue());
            System.out.println("------------------------------");
        
        
    }
    
    
   
    
    /**
     * Affiche les attributs dépassant un seuil avec leurs mesures
     * 
     * @param inst Les instances du dataset
     * @param threshold Le seuil à dépasser
     * @throws Exception Si le calcul des mesures échoue
     */
    public void printAttributesAboveThreshold(Instances inst, double threshold) throws Exception {
        // Calculer les mesures pour chaque attribut
        double[] featureMeasures = computeFeatureMeasures(inst);

        // Créer une liste de tous les attributs avec leurs mesures (sauf la classe)
        ArrayList<Pair<Attribute, Double>> allAttributeMeasures = new ArrayList<>();
        for (int i = 0; i < inst.numAttributes(); i++) {
            if (i != inst.classIndex()) {
                allAttributeMeasures.add(new Pair<>(inst.attribute(i), featureMeasures[i]));
            }
        }

        // Trier tous les attributs selon la métrique
        allAttributeMeasures.sort((a1, a2) -> {
            if (FeatureMeasureMethod == FEAT_ENTROPY) {
                return Double.compare(a1.getValue(), a2.getValue()); // Ascendant
            } else {
                return Double.compare(a2.getValue(), a1.getValue()); // Descendant
            }
        });

        // === Affichage de TOUS les attributs ===
        System.out.println("\n▬▬▬▬▬▬▬▬▬▬▬▬▬ ALL ATTRIBUTES ▬▬▬▬▬▬▬▬▬▬▬▬▬");
        System.out.println("[" + featureMeasureToString(FeatureMeasureMethod) + "]\n");
        System.out.println(String.format("%-6s%-30s%12s", "Rang", "Attribut", "Valeur"));
        System.out.println("-------------------------------------------");
        int rank = 1;
        for (Pair<Attribute, Double> p : allAttributeMeasures) {
            String cleanAttrName = p.getKey().name().trim().replaceAll("\\s+", " ");
            System.out.println(String.format("%-6d%-30s%12.4f",
                rank++,
                cleanAttrName.length() > 30 ? cleanAttrName.substring(0, 27) + "..." : cleanAttrName,
                p.getValue()));
        }
        System.out.println("-------------------------------------------");

        // Statistiques de tous les attributs
        double max = allAttributeMeasures.get(0).getValue();
        double min = allAttributeMeasures.get(allAttributeMeasures.size() - 1).getValue();
        double average = allAttributeMeasures.stream()
            .mapToDouble(Pair::getValue)
            .average()
            .orElse(0);

        System.out.println("\n📊 Statistics:");
        System.out.println(String.format("\nMax: %8.4f  Min: %8.4f  Moy: %8.4f", max, min, average));
        // Obtenir les attributs dépassant le seuil
        ArrayList<Attribute> selectedAttributes = selectAttributesAboveThreshold(inst, threshold);
        if (selectedAttributes.isEmpty()) {
            System.out.println("\n ⚠ No attributes above threshold: " + threshold);
            return;
        }

        // Créer la liste pour les affichages filtrés
        ArrayList<Pair<Attribute, Double>> attributeMeasures = new ArrayList<>();
        for (Attribute attr : selectedAttributes) {
            int index = attr.index();
            attributeMeasures.add(new Pair<>(attr, featureMeasures[index]));
        }

        // Trier cette liste aussi
        attributeMeasures.sort((a1, a2) -> {
            if (FeatureMeasureMethod == FEAT_ENTROPY) {
                return Double.compare(a1.getValue(), a2.getValue());
            } else {
                return Double.compare(a2.getValue(), a1.getValue());
            }
        });

        // === Affichage des attributs au-dessus du seuil ===
        System.out.println("\n▬▬▬▬▬▬▬▬ ATTRIBUTES ABOVE THRESHOLD (" + threshold + ") ▬▬▬▬▬▬▬▬");
        System.out.println("[" + featureMeasureToString(FeatureMeasureMethod) + "]\n");
        System.out.println("Number of selected attributes: " + selectedAttributes.size() + "\n");

        System.out.println(String.format("%-6s%-30s%12s", "Rang", "Attribut", "Valeur"));
        System.out.println("-------------------------------------------");

        for (int i = 0; i < attributeMeasures.size(); i++) {
            Pair<Attribute, Double> p = attributeMeasures.get(i);
            String cleanAttrName = p.getKey().name().trim().replaceAll("\\s+", " ");
            System.out.println(String.format("%-6d%-30s%12.4f",
                i + 1,
                cleanAttrName.length() > 30 ? cleanAttrName.substring(0, 27) + "..." : cleanAttrName,
                p.getValue()));
        }

       
    }

    private static String centerText(String text, int width) {
        if (text.length() >= width) return text;
        int padding = (width - text.length()) / 2;
        return String.format("%" + (padding + text.length()) + "s", text);
    }

    private static String truncate(String text, int maxLength) {
        return text.length() > maxLength ? text.substring(0, maxLength-3) + "..." : text;
    }
    // Classe utilitaire Pair pour stocker les paires attribut-mesure
    private static class Pair<K, V> {
        private final K key;
        private final V value;

        public Pair(K key, V value) {
            this.key = key;
            this.value = value;
        }

        public K getKey() { return key; }
        public V getValue() { return value; }
    }

   


    private ArrayList<String> computeFermeture(Instances inst, ArrayList<Integer> instances) {
        ArrayList<String> fermeture = new ArrayList<>();
        
        // Initialiser avec -null- pour tous les attributs
        for (int i = 0; i < inst.numAttributes() - 1; i++) {
            fermeture.add("-null-");
        }
        
        for (int attrIdx = 0; attrIdx < inst.numAttributes() - 1; attrIdx++) {
            if (attrIdx == inst.classIndex()) continue;
            
            String firstValue = inst.instance(instances.get(0)).stringValue(attrIdx);
            boolean isConstant = true;
            
            for (int instIdx : instances) {
                if (!inst.instance(instIdx).stringValue(attrIdx).equals(firstValue)) {
                    isConstant = false;
                    break;
                }
            }
            
            if (isConstant) {
                int fermetureIdx = (attrIdx < inst.classIndex()) ? attrIdx : attrIdx - 1;
                fermeture.set(fermetureIdx, firstValue);
            }
        }
        
        return fermeture;
    }
    

    
  
    
    
    
    private ArrayList<String> computeMultiAttributeClosure(Instances inst, 
            ArrayList<Integer> instances,
            ArrayList<Attribute> selectedAttributes) {
			ArrayList<String> closure = new ArrayList<>();
			// Initialiser avec -null- pour tous les attributs
			for (int i = 0; i < inst.numAttributes() - 1; i++) {
				closure.add("-null-");
			}
			
			// Vérifier seulement les attributs sélectionnés
			for (Attribute attr : selectedAttributes) {
				String firstValue = inst.instance(instances.get(0)).stringValue(attr);
				boolean isConstant = true;
			
				for (int instIdx : instances) {
					if (!inst.instance(instIdx).stringValue(attr).equals(firstValue)) {
						isConstant = false;
						break;
					}
			}
			
					if (isConstant) {
						int closureIdx = (attr.index() < inst.classIndex()) ? attr.index() : attr.index() - 1;
						closure.set(closureIdx, firstValue);
					}
			}
			
			return closure;
			}
	    
    
    
    

    private ArrayList<Integer> getInstancesForValue(Instances inst, Attribute attr, int valueIdx) {
        ArrayList<Integer> instances = new ArrayList<>();
        for (int i = 0; i < inst.numInstances(); i++) {
            if ((int)inst.instance(i).value(attr) == valueIdx) {
                instances.add(i);
            }
        }
        return instances;
    }

    

// Interface pour les différentes stratégies d'extraction de règles
interface RuleExtractionStrategy {
    ArrayList<Classification_Rule> extractRules(Instances inst, Attribute bestAttribute, double[] featureMeasures) throws Exception;
}

// Fermeture de la valeur la plus pertinente
class BestValueClosureStrategy implements RuleExtractionStrategy {
    private boolean m_Debug;
    private int Policy_Selection;
    private RuleExtractor ruleExtractor;
    
    public BestValueClosureStrategy(boolean debug, int policySelection, RuleExtractor ruleExtractor) {
        this.m_Debug = debug;
        this.Policy_Selection = policySelection;
        this.ruleExtractor = ruleExtractor;
    }


    @Override
    public ArrayList<Classification_Rule> extractRules(Instances inst, Attribute bestAttribute, double[] featureMeasures) throws Exception {
        ArrayList<Classification_Rule> rules = new ArrayList<>();
        
        if (m_Debug)
            System.out.println("Generating CNC in progress...");

        // Find the best distinct value based on support
        int supportDistVal = 0;
        int indexBestDistVal = 0;
        int suppBestDistVal = 0;

        // To track ties
        ArrayList<Integer> bestIndexes = new ArrayList<>();

        for (int i = 0; i < inst.numDistinctValues(bestAttribute.index()); i++) {
            supportDistVal = 0;
            String currentVal = inst.attribute(bestAttribute.index()).value(i);

            for (int j = 0; j < inst.numInstances(); j++) {
                if (inst.instance(j).stringValue(bestAttribute.index()).equals(currentVal)) {
                    supportDistVal++;
                }
            }

            if (m_Debug) {
                System.out.println("- Value \"" + currentVal + "\" has support of: " + supportDistVal);
            }

            if (supportDistVal > suppBestDistVal) {
                suppBestDistVal = supportDistVal;
                indexBestDistVal = i;
                bestIndexes.clear();
                bestIndexes.add(i);
            } else if (supportDistVal == suppBestDistVal) {
                bestIndexes.add(i); // tied values
                if (Policy_Selection == RuleExtractor.POLICY_LAST) {
                    indexBestDistVal = i; // keep the last one
                } else if (Policy_Selection == RuleExtractor.POLICY_FIRST) {
                    // keep the first one already selected
                }
            }
        }

        // Display final best value
        if (m_Debug) {
            System.out.println();

            if (bestIndexes.size() > 1) {
                System.out.print("Warning: Multiple values with maximum support (" + suppBestDistVal + "): ");
                for (int idx : bestIndexes) {
                    System.out.print("\"" + bestAttribute.value(idx) + "\" ");
                }
                System.out.println();

                if (Policy_Selection == RuleExtractor.POLICY_LAST) {
                    System.out.println("Selection policy: LAST → selecting the last one: \"" + bestAttribute.value(indexBestDistVal) + "\"\n");
                } else if (Policy_Selection == RuleExtractor.POLICY_FIRST) {
                    System.out.println("Selection policy: FIRST → selecting the first one: \"" + bestAttribute.value(bestIndexes.get(0)) + "\"\n");
                }
            } else {
                System.out.println("Unique best value: \"" + bestAttribute.value(indexBestDistVal) + "\" with support of: " + suppBestDistVal + "\n");
            }

            System.out.println("Selected attribute for closure calculation: " + bestAttribute.name() + 
                               " (index: " + bestAttribute.index() + ")");
        }
        
        // Find instances with this attribute value
        ArrayList<Integer> instances = new ArrayList<>();
        for(int i=0; i<inst.numInstances(); i++) {
            if(inst.instance(i).stringValue(bestAttribute.index()).equals(bestAttribute.value(indexBestDistVal))) {
                instances.add(i);
            }
        }
        int numInstances = instances.size();
        
        // Calculate closure attributes
        ArrayList<String> attributes = new ArrayList<>();
        String nullValue = "-null-";
        for(int i=0; i<inst.numAttributes()-1; i++) {
            int count = 0;
            String firstDistVal = inst.instance(instances.get(0)).stringValue(i);
            
            for(int j=0; j<numInstances;) {
                if(inst.instance(instances.get(j)).stringValue(i).equals(firstDistVal)) {
                    count++;
                    if(count == numInstances) {
                        attributes.add(inst.instance(instances.get(0)).stringValue(i));
                    }
                    j++;
                } else {
                    j = numInstances;
                    attributes.add(nullValue);
                }
            }
        }
        
        // Display Galois closure using RuleExtractor reference
        if (m_Debug && ruleExtractor != null) {
            ruleExtractor.printGaloisClosure(instances, attributes, inst);
        }
        
        if(m_Debug) {
            System.out.print("\nList of selected attribute values:  ");
            for(int i=0; i<attributes.size()-1; i++) {
                if(!attributes.get(i).equals(nullValue)) {
                    System.out.print(attributes.get(i) + ", ");
                }
            }
        }
        
        // Find majority class
        int majorityClassIndex = findMajorityClass(inst, instances);
        
        if(m_Debug)
            System.out.println("\n\nMajority class is: " + inst.attribute(inst.classIndex()).value(majorityClassIndex));
        
        // Create rule
        Classification_Rule rule = new Classification_Rule(
            attributes,
            majorityClassIndex,
            inst.attribute(inst.classIndex()).value(majorityClassIndex),
            featureMeasures[Utils.maxIndex(featureMeasures)]
        );
        
        // Add attribute names to rule
        ArrayList<String> attributeNames = new ArrayList<>();
        for (int i = 0; i < inst.numAttributes() - 1; i++) {
            attributeNames.add(inst.attribute(i).name());
        }
        rule.setAttributeNames(attributeNames);
        
        if (m_Debug) {
            System.out.println("Rule: " + rule.affich_nom_rule(true));
        }
        
        rules.add(rule);
        return rules;
    }

    
    // Méthode pour trouver la classe majoritaire
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
}

// Fermeture basée sur les classes
class ClassBasedClosureStrategy implements RuleExtractionStrategy {
    private boolean m_Debug;
    private int Policy_Selection;
    private RuleExtractor ruleExtractor;
    
    public ClassBasedClosureStrategy(boolean debug, int policySelection, RuleExtractor ruleExtractor) {
        this.m_Debug = debug;
        this.Policy_Selection = policySelection;
        this.ruleExtractor = ruleExtractor;
    }
    
    @Override
   public ArrayList<Classification_Rule> extractRules(Instances inst, Attribute bestAttribute, double[] featureMeasures) throws Exception {
    ArrayList<Classification_Rule> rules = new ArrayList<>();
    
    if(m_Debug)
        System.out.println("\nGenerating a CNC from class-based closure for the selected attribute");
    
    // For each class
    for (int classIdx = 0; classIdx < inst.numClasses(); classIdx++) {
        if (m_Debug) {
            System.out.println("\nProcessing class: " + inst.attribute(inst.classIndex()).value(classIdx));
        }
        
        // Collect instances of this class
        ArrayList<Integer> classInstances = new ArrayList<>();
        for (int i = 0; i < inst.numInstances(); i++) {
            if (inst.instance(i).classValue() == classIdx) {
                classInstances.add(i);
            }
        }
        
        if (classInstances.isEmpty()) continue;
        
        // Find most frequent attribute value for this class
        int[] valueCounts = new int[bestAttribute.numValues()];
        for (int i : classInstances) {
            int valueIdx = (int) inst.instance(i).value(bestAttribute.index());
            valueCounts[valueIdx]++;
        }
        
        int majorityValueIdx = 0;
        for (int i = 0; i < valueCounts.length; i++) {
            if (valueCounts[i] > valueCounts[majorityValueIdx]) {
                majorityValueIdx = i;
            } else if (valueCounts[i] == valueCounts[majorityValueIdx] && Policy_Selection == RuleExtractor.POLICY_LAST) {
                majorityValueIdx = i;
            }
        }
        
        // Collect instances with this attribute value
        ArrayList<Integer> relevantInstances = new ArrayList<>();
        for (int i : classInstances) {
            if (inst.instance(i).value(bestAttribute.index()) == majorityValueIdx) {
                relevantInstances.add(i);
            }
        }
        
        if (relevantInstances.isEmpty()) continue;
        
        // Compute attribute closure
        ArrayList<String> attributes = computeAttributeClosure(inst, relevantInstances);
        
        // Display Galois closure using RuleExtractor reference
        if (m_Debug && ruleExtractor != null) {
            ruleExtractor.printGaloisClosure(relevantInstances, attributes, inst);
        }
        
        // Create rule
        Classification_Rule rule = new Classification_Rule(
            attributes,
            classIdx,
            inst.attribute(inst.classIndex()).value(classIdx),
            featureMeasures[Utils.maxIndex(featureMeasures)]  /* weight */
        );
        
        // Add attribute names to rule
        ArrayList<String> attributeNames = new ArrayList<>();
        for (int i = 0; i < inst.numAttributes() - 1; i++) {
            attributeNames.add(inst.attribute(i).name());
        }
        rule.setAttributeNames(attributeNames);
        
        rules.add(rule);
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
}

// Fermeture pour toutes les valeurs
class AllValuesClosureStrategy implements RuleExtractionStrategy {
    private boolean m_Debug;
    private RuleExtractor ruleExtractor;
    
    public AllValuesClosureStrategy(boolean debug, int policySelection, RuleExtractor ruleExtractor) {
        this.m_Debug = debug;
        this.ruleExtractor = ruleExtractor;
    }
    
    @Override
    public ArrayList<Classification_Rule> extractRules(Instances inst, Attribute bestAttribute, double[] featureMeasures) throws Exception {
    ArrayList<Classification_Rule> rules = new ArrayList<>();
    
    if(m_Debug)
        System.out.println("\nGenerating a CNC from closure of all values of the selected attribute");
    
    // For each attribute value
    for (int valueIdx = 0; valueIdx < bestAttribute.numValues(); valueIdx++) {
        if (m_Debug) {
            System.out.println("\nProcessing value: " + bestAttribute.value(valueIdx));
        }
        
        // Find instances with this value
        ArrayList<Integer> instances = new ArrayList<>();
        for (int i = 0; i < inst.numInstances(); i++) {
            if (inst.instance(i).value(bestAttribute.index()) == valueIdx) {
                instances.add(i);
            }
        }
        
        if (instances.isEmpty()) continue;
        
        // Compute attribute closure
        ArrayList<String> attributes = computeAttributeClosure(inst, instances);
        
        // Display Galois closure using RuleExtractor reference
        if (m_Debug && ruleExtractor != null) {
            ruleExtractor.printGaloisClosure(instances, attributes, inst);
        }
        
        // Find majority class
        int majorityClassIndex = findMajorityClass(inst, instances);
        
        // Create rule
        Classification_Rule rule = new Classification_Rule(
            attributes,
            majorityClassIndex,
            inst.attribute(inst.classIndex()).value(majorityClassIndex),
            featureMeasures[Utils.maxIndex(featureMeasures)]
        );
        
        // Add attribute names to rule
        ArrayList<String> attributeNames = new ArrayList<>();
        for (int i = 0; i < inst.numAttributes() - 1; i++) {
            attributeNames.add(inst.attribute(i).name());
        }
        rule.setAttributeNames(attributeNames);
        
        // Calculate weight
        double weight = calculateRuleWeight(inst, rule);
        rule.setRule_Ponderation(weight);
        
        // Display created rule
        if (m_Debug) {
            System.out.println("Rule: " + rule.affich_nom_rule(true));
        }
        
        rules.add(rule);
    }
    
    return rules;
}
    // Méthode pour trouver la classe majoritaire
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
    
    // Méthode pour calculer la fermeture des attributs
    private ArrayList<String> computeAttributeClosure(Instances inst, ArrayList<Integer> instances) {
        ArrayList<String> attributes = new ArrayList<>();
        
        if (instances.isEmpty()) {
            // Cas où il n'y a pas d'instances, on retourne des valeurs nulles
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
}

public static void main (String []args) throws Exception {
	boolean debug=true;
	RuleExtractor exttractor= new RuleExtractor(debug,RuleExtractor.FEAT_INFOGAIN,RuleExtractor.POLICY_LAST,RuleExtractor.CONCEPT_LEARNING_ABOVE_THRESHOLD_ALL);
	// 1. Charger les données (Instances inst)
	String dataPath="C:\\Program Files\\Weka-3-8-6\\data\\weather.nominal.arff";
	Instances inst=null;
	try {
		inst = DataSource.read(dataPath);
	} catch (Exception e) {
		// TODO Auto-generated catch block
		e.printStackTrace();
	}
    inst.setClassIndex(inst.numAttributes() - 1);
	// 2. Sélectionner les attributs pertinents
    exttractor.printAttributesUnderThreshold(inst, 0.1);
    ArrayList<Classification_Rule>  rules = exttractor.extractAttributeUnderThresholdMajorityValues
    		(inst,0.1);

	

}

}