package weka.classifiers.fca;

import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Date;
import java.util.HashMap;
import java.util.Map;
import weka.attributeSelection.*;
import weka.core.Attribute;
import weka.core.Instances;
import weka.core.Utils;
import weka.core.converters.ConverterUtils.DataSource;

	/**
 * Classe principale qui gère l'extraction des règles avec support pour les connexions de Galois
 */
public class RuleExtractor {
    private boolean m_Debug;
    private int FeatureMeasureMethod;
    private int Policy_Selection;
    private InfoGainAttributeEval m_InfoGainAttributeEval;
    private GainRatioAttributeEval m_GainRatioAttributeEval;
	public _GaloisConnection m_GaloisConnection;
	
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
        this.m_GaloisConnection= new _GaloisConnection(this.m_Debug);
    
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
    public double[] computeFeatureMeasures(Instances inst) throws Exception {
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
	 * Ancienne méthode extractAllAttributesAllValues transformée pour appeler generateConcepts
	 * tout en conservant sa logique originale
	 */
	public ArrayList<Classification_Rule> extractAllAttributesAllValues(Instances inst) throws Exception {
	    ArrayList<Attribute> attributes = new ArrayList<>();
	    for (int i = 0; i < inst.numAttributes() - 1; i++) {
	        attributes.add(inst.attribute(i));
	    }
	    
	    ArrayList<String> allConceptsList = new ArrayList<>();
	    boolean debug = m_Debug;
	    
	    if (debug) {
	        System.out.println("\n=== EXTRACTION AVEC TOUS LES ATTRIBUTS ET VALEURS ===");
	        System.out.println("Nombre d'attributs à traiter: " + attributes.size());
	    }
	    
	    // Appel à la nouvelle méthode generateConcepts
	    ArrayList<Classification_Rule> rules = this.m_GaloisConnection.generateConcepts(inst, attributes, allConceptsList, debug);
	    
	    if (debug) {
	        System.out.println("\n=== LISTE COMPLÈTE DES CONCEPTS GÉNÉRÉS ===");
	        System.out.println("Nombre total de règles: " + rules.size());
	        System.out.println("--------------------------------------------");
	    }
	    
	    return rules;
	}
 
    
    /**
     * Extrait des règles en explorant toutes les combinaisons possibles d'attributs
     * et en calculant les fermetures pour chaque combinaison.
     * 
     * @param inst Le jeu de données à analyser
     * @return Liste des règles de classification générées
     * @throws Exception en cas d'erreur
     */
   
    public ArrayList<Classification_Rule> extractAllAttributesAllValuesAllCombinations(Instances inst) throws Exception {
    // 1. Préparer la liste des attributs (sans la classe)
    ArrayList<Attribute> attributes = new ArrayList<>();
    for (int i = 0; i < inst.numAttributes() - 1; i++) {
        attributes.add(inst.attribute(i));
    }

    // 2. Générer les concepts
    ArrayList<String> allConcepts = new ArrayList<>();
    ArrayList<Classification_Rule> rules = this.m_GaloisConnection.generateConceptsAllCombinations(inst, attributes, allConcepts, m_Debug);

    
    

    return rules;
}

  
   /**
     * Extrait des règles en utilisant tous les attributs mais seulement la valeur majoritaire all attributes Maj value
     */
    public ArrayList<Classification_Rule> extractAllAttributesMajorityValue(Instances inst) throws Exception {
        ArrayList<Classification_Rule> rules = new ArrayList<>();
        ArrayList<Attribute> attributes= new ArrayList<>();
        ArrayList<String> allConcepts= new ArrayList<>();
        
        // Pour chaque attribut
        for (int attrIdx = 0; attrIdx < inst.numAttributes() - 1; attrIdx++) {
        	attributes.add(inst.attribute(attrIdx)); 
        }
        rules= this.m_GaloisConnection.generateConceptsFromMajorityValues(inst, attributes, allConcepts, Policy_Selection, m_Debug);

        
        return rules;
    }
       
    
    /**
     * Extrait des règles en utilisant toutes les combinaisons d'attributs possibles
     * avec seulement leurs valeurs majoritaires
     */
    public ArrayList<Classification_Rule> extractAllAttributesMajorityValueAllCombinations(Instances inst) throws Exception {
        // 1. Pré-calculer les valeurs majoritaires pour chaque attribut
        Map<Attribute, String> majorityValues = new HashMap<>();
        for (int attrIdx = 0; attrIdx < inst.numAttributes() - 1; attrIdx++) {
            Attribute attr = inst.attribute(attrIdx);
            majorityValues.put(attr, findMajorityValue(inst, attrIdx));
        }

        // 2. Préparer la liste des attributs (sans la classe)
        ArrayList<Attribute> attributes = new ArrayList<>();
        for (int i = 0; i < inst.numAttributes() - 1; i++) {
            attributes.add(inst.attribute(i));
        }
        ArrayList<String> allConcepts = new ArrayList<>();

        // 3. Générer les concepts à partir des valeurs majoritaires
        ArrayList<Classification_Rule> rules=this.m_GaloisConnection.generateConceptsFromMajorityValuesAllCombinations(inst, attributes, majorityValues,allConcepts,m_Debug);
        
	    
        return rules;
    }

	 /**
	   * Méthode pour extraire des règles basées sur l attribut pertinant  et tout ses valeurs
	   */
    public ArrayList<Classification_Rule> extractPertinentAttributeAllValues(Instances inst) throws Exception{


    	// Initialiser les évaluateurs d'attributs selon la méthode choisie
        initializeFeatureEvaluator(inst);
        
        // Calculer les mesures pour chaque attribut
        double[] featureMeasures = computeFeatureMeasures(inst);
        Attribute bestAttribute = this.selectBestAttribute(inst, featureMeasures);
        ArrayList<Attribute> attributes= new ArrayList<>();
        attributes.add(bestAttribute);
        ArrayList<String> allConceptsList = new ArrayList<>();
	    boolean debug = m_Debug;
	    
	    if (debug) {
	        System.out.println("\n=== EXTRACTION AVEC L'ATTRIBUTS PERTINENT ET TOUS SES VALEURS ===");
	    }
	    
	    // Appel à la nouvelle méthode generateConcepts
	    ArrayList<Classification_Rule> rules = this.m_GaloisConnection.generateConcepts(inst, attributes, allConceptsList, debug);
	    
	    if (debug) {
	        System.out.println("\n=== LISTE COMPLÈTE DES CONCEPTS GÉNÉRÉS ===");
	        System.out.println("Nombre total de règles: " + rules.size());
	        System.out.println("--------------------------------------------");
	    }
	    
	    return rules;
        

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
	        threshold=checkThreshold(threshold,inst,featureMeasures);
	            

	        ArrayList<Attribute> topAttributes =  selectAttributesAboveThreshold(inst,  threshold);
	
	        ArrayList<String> allConceptsList = new ArrayList<>();
		    boolean debug = m_Debug;
		    
		  
		    // Appel à la nouvelle méthode generateConcepts
		    rules = this.m_GaloisConnection.generateConcepts(inst, topAttributes, allConceptsList, debug);
		    
		    if (debug) {
		        System.out.println("\n=== LISTE COMPLÈTE DES CONCEPTS GÉNÉRÉS ===");
		        System.out.println("Nombre total de règles: " + rules.size());
		        System.out.println("--------------------------------------------");
		    }
		    
		    return rules;
	}
    
	 
	/**
	* Extrait des règles basées sur toutes les combinaisons d'attributs dépassant le seuil
	*/
	public ArrayList<Classification_Rule> extractAttributeAboveThresholdAllValuesAllCombinations(Instances inst, double threshold) throws Exception {
		    ArrayList<Classification_Rule> rules = new ArrayList<>();
		    double[] featureMeasures = computeFeatureMeasures(inst);
		    threshold = checkThreshold(threshold, inst, featureMeasures);
		    
		    ArrayList<Attribute> topAttributes = selectAttributesAboveThreshold(inst, threshold);
		
		    if (m_Debug) {
		        System.out.println("\n=== Processing attribute combinations above threshold ===");
		        System.out.println("Selected attributes count: " + topAttributes.size());
		        System.out.println("Significance threshold: " + threshold);
		    }
		    
		    
	        ArrayList<String> allConcepts = new ArrayList<>();
	        rules = this.m_GaloisConnection.generateConcepts(inst, topAttributes, allConcepts,m_Debug);

		    
		    
		    
		    return rules;
		}
		
	
	  /**
	     * Méthode pour extraire des règles basées sur les attributs depassant le seuil  et sa val maj
	     */
	 public ArrayList<Classification_Rule> extractAttributeAboveThresholdMajorityValues(Instances inst, double threshold) throws Exception {
		    ArrayList<Classification_Rule> rules = new ArrayList<>();
		    double[] featureMeasures = computeFeatureMeasures(inst);
	        threshold=checkThreshold(threshold,inst,featureMeasures);

		    ArrayList<Attribute> topAttributes = selectAttributesAboveThreshold(inst, threshold);

		    ArrayList<String> allConcepts= new ArrayList<>();
	        
		      
	        rules= this.m_GaloisConnection.generateConceptsFromMajorityValues(inst, topAttributes, allConcepts, Policy_Selection, m_Debug);

		    
		    return rules;
		}
    
	  
	 public ArrayList<Classification_Rule> extractAttributeAboveThresholdMajorityValuesAllCombinations(Instances inst, double threshold) throws Exception {
		    
		    ArrayList<Attribute> topAttributes = selectAttributesAboveThreshold(inst, threshold);

		  // 1. Pré-calculer les valeurs majoritaires pour chaque attribut
	        Map<Attribute, String> majorityValues = new HashMap<>();
		    for (Attribute attr : topAttributes) {
		        majorityValues.put(attr, findMajorityValue(inst, attr.index()));
		    }
	        ArrayList<String> allConcepts = new ArrayList<>();


	        // 3. Générer les concepts à partir des valeurs majoritaires
		    ArrayList<Classification_Rule> rules = this.m_GaloisConnection.generateConceptsFromMajorityValuesAllCombinations(inst, topAttributes, majorityValues,allConcepts,m_Debug);
	        
		    
		    return rules;
		  
		}

	  /**
	     * Méthode pour extraire des règles basées sur les attributs depassant le seuil  et tous ses valeur
	     */
	 public ArrayList<Classification_Rule> extractAttributeUnderThresholdAllValues(Instances inst, double threshold) throws Exception {
			// utilisation de methode compute multiAtt closure
	        ArrayList<Classification_Rule> rules = new ArrayList<>();
		    double[] featureMeasures = computeFeatureMeasures(inst);

	        threshold=checkThreshold(threshold,inst,featureMeasures);

	        ArrayList<Attribute> topAttributes =  selectAttributesUnderThreshold(inst,  threshold);
	    	
	        ArrayList<String> allConceptsList = new ArrayList<>();
		    boolean debug = m_Debug;
		    
		  
		    // Appel à la nouvelle méthode generateConcepts
		    rules = this.m_GaloisConnection.generateConcepts(inst, topAttributes, allConceptsList, debug);
		    
		    
		    return rules;
	    }
	 
		  /**
			 * Extrait des règles basées sur toutes les combinaisons d'attributs dépassant le seuil
			 */
	 public ArrayList<Classification_Rule> extractAttributeUnderThresholdAllValuesAllCombinations(Instances inst, double threshold) throws Exception {
			    ArrayList<Classification_Rule> rules = new ArrayList<>();
			    double[] featureMeasures = computeFeatureMeasures(inst);
			    threshold = checkThreshold(threshold, inst, featureMeasures);
			    
			    ArrayList<Attribute> topAttributes = selectAttributesUnderThreshold(inst, threshold);
			
			    if (m_Debug) {
			        System.out.println("\n=== Processing attribute combinations above threshold ===");
			        System.out.println("Selected attributes count: " + topAttributes.size());
			        System.out.println("Significance threshold: " + threshold);
			    }
			    Map<Attribute, String> majorityValues = new HashMap<>();
			    for (Attribute attr : topAttributes) {
			        majorityValues.put(attr, findMajorityValue(inst, attr.index()));
			    }
			
			    ArrayList<String> allConcepts = new ArrayList<>();
		        rules = this.m_GaloisConnection.generateConceptsAllCombinations(inst, topAttributes, allConcepts, m_Debug);

			    
			    
			    return rules;
			}
				  	
	  	
	  	/**
	     * Méthode pour extraire des règles basées sur les attributs depassant le seuil  et tous ses valeur
	     */
	public ArrayList<Classification_Rule> extractAttributeUnderThresholdMajorityValues(Instances inst, double threshold) throws Exception {
				ArrayList<Classification_Rule> rules = new ArrayList<>();
			    double[] featureMeasures = computeFeatureMeasures(inst);
		        threshold=checkThreshold(threshold,inst,featureMeasures);
			    ArrayList<Attribute> topAttributes = selectAttributesUnderThreshold(inst, threshold);

			    ArrayList<String> allConcepts= new ArrayList<>();
		        
		      
		        rules= this.m_GaloisConnection.generateConceptsFromMajorityValues(inst, topAttributes, allConcepts, Policy_Selection, m_Debug);

		        
			    
			    return rules;
			}
  
	public ArrayList<Classification_Rule> extractAttributeUnderThresholdMajorityValuesAllCombinations(Instances inst, double threshold) throws Exception {
			    ArrayList<Classification_Rule> rules = new ArrayList<>();
			    double[] featureMeasures = computeFeatureMeasures(inst);
			    threshold = checkThreshold(threshold, inst, featureMeasures);
			    
			    // 1. Sélection des attributs au-dessus du seuil et calcul de leurs valeurs majoritaires
			    ArrayList<Attribute> topAttributes = selectAttributesUnderThreshold(inst, threshold);
			    Map<Attribute, String> majorityValues = new HashMap<>();
			    for (Attribute attr : topAttributes) {
			        majorityValues.put(attr, findMajorityValue(inst, attr.index()));
			    }

		        ArrayList<String> allConcepts = new ArrayList<>();

		        // 3. Générer les concepts à partir des valeurs majoritaires
			    rules = this.m_GaloisConnection.generateConceptsFromMajorityValuesAllCombinations(inst, topAttributes, majorityValues,allConcepts,m_Debug);
		        
			    return rules;
			}
		
		
	
	public double checkThreshold(double threshold, Instances inst,double[] featureMeasures  ) {
			  if (threshold == -1) {
				    ArrayList<Pair<Attribute, Double>> allAttributeMeasures = new ArrayList<>();
				    for (int i = 0; i < inst.numAttributes(); i++) {
				        if (i != inst.classIndex()) {
				            allAttributeMeasures.add(new Pair<>(inst.attribute(i), featureMeasures[i]));
				        }
				    }

				    threshold = allAttributeMeasures.stream()
				        .mapToDouble(Pair::getValue)
				        .average()
				        .orElse(0);

				    if (m_Debug) {
				        System.out.println("\n⚠ Auto-adjusted threshold: No user-defined value provided. Setting threshold to the average feature score: " + threshold);
				    }
				    
				}
			  return threshold;
		  }
	    
		

    
    private String findMajorityValue(Instances inst, int attrIdx) {
        Map<String, Integer> valueCounts = new HashMap<>();
        
        for (int i = 0; i < inst.numInstances(); i++) {
            String value = inst.instance(i).stringValue(attrIdx);
            valueCounts.put(value, valueCounts.getOrDefault(value, 0) + 1);
        }
        
        return valueCounts.entrySet().stream()
                         .max(Map.Entry.comparingByValue())
                         .map(Map.Entry::getKey)
                         .orElse("-null-");
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
        System.out.println("━".repeat(80));
        
        int rank = 1;
        for (Pair<Attribute, Double> p : allAttributeMeasures) {
            String cleanAttrName = p.getKey().name().trim().replaceAll("\\s+", " ");
            System.out.println(String.format("%-6d%-30s%12.4f",
                rank++,
                cleanAttrName.length() > 30 ? cleanAttrName.substring(0, 27) + "..." : cleanAttrName,
                p.getValue()));
        }
        
        System.out.println("--------------------------------------------------------------------------------------");

        
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
        System.out.printf("\n▬▬▬▬▬▬▬▬ ATTRIBUTES UNDER THRESHOLD (Threshold=%.4f)  ▬▬▬▬▬▬▬▬", threshold);
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
                System.out.println("⚠No attributes available!");
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
        System.out.printf("\n▬▬▬▬▬▬▬▬ ATTRIBUTES ABOVE THRESHOLD  (Threshold=%.4f)  ▬▬▬▬▬▬▬▬", threshold);
        System.out.println("[" + featureMeasureToString(FeatureMeasureMethod) + "]\n");
        System.out.println("Number of selected attributes: " + selectedAttributes.size() + "\n");

        System.out.println(String.format("%-6s%-30s%12s", "Rank", "Attribute", "Value"));
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

	

}

}