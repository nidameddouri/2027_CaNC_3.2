package weka.classifiers.fca;

import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Date;

import weka.attributeSelection.*;
import weka.core.Attribute;
import weka.core.Instances;
import weka.core.Utils;

import java.util.Enumeration;

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
    
    // Concept Learning Strategy
    public static final int CONCEPT_LEARNING_PERTINENT = 1;
    public static final int CONCEPT_LEARNING_ALL = 2;
    
    // Constantes pour les stratégies Galois
    public static final int GALOIS_PERTINENT_MAJORITY = 1;  // Pertinent attribute & Majority Value
    public static final int GALOIS_PERTINENT_ALL = 2;       // Pertinent Attribute & All Values
    public static final int GALOIS_ALL_MAJORITY = 3;        // All Attributes & Majority Value
    public static final int GALOIS_ALL_ALL = 4;             // All Attributes & All Values
    
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
                + "ConceptLearning: " + conceptLearningToString(conceptLearningStrategy) + ",\n"
                + "FeatureMeasureMethod: " + featureMeasureToString(featureMeasureMethod) + ",\n"
                + "Policy_Selection: " + policySelectionToString(policySelection));
        }
    }


    /**
     * Méthode pour convertir les stratégies de concept learning en chaînes
     */
    private String conceptLearningToString(int cl) {
        switch (cl) {
            case 1: return "Closure of pertinent attribute";
            case 2: return "Closure of ALL attributes";
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

        System.out.println("\n--- CALCUL DE LA FERMETURE DE GALOIS ---\n");
        
        StringBuilder sb = new StringBuilder();
        
        // Afficher l'ensemble des objets (instances)
        sb.append("Fermeture de Galois : ({");
        boolean first = true;
        for (Integer objIdx : objects) {
            if (!first) {
                sb.append(", ");
            }
            sb.append("o").append(objIdx + 1);
            first = false;
        }
        sb.append("}, {");
        
        // Afficher l'ensemble des attributs non-null
        first = true;
        for (int i = 0; i < attributes.size(); i++) {
            if (!attributes.get(i).equals("-null-")) {
                if (!first) {
                    sb.append(", ");
                }
                sb.append(inst.attribute(i).name()).append("=").append(attributes.get(i));
                first = false;
            }
        }
        sb.append("})");
        
        System.out.println(sb.toString());
        
        // Afficher les instances correspondantes
        System.out.println("\nInstances correspondantes :");
        for (Integer objIdx : objects) {
            StringBuilder instanceStr = new StringBuilder();
            for (int i = 0; i < inst.numAttributes(); i++) {
                if (i > 0) instanceStr.append(", ");
                instanceStr.append(inst.attribute(i).name()).append("=")
                           .append(inst.instance(objIdx).stringValue(i));
            }
            System.out.println("  o" + (objIdx + 1) + " : " + instanceStr);
        }
        
        // Afficher les attributs correspondants
        System.out.println("\nAttributs correspondants :");
        for (int i = 0; i < attributes.size(); i++) {
            if (!attributes.get(i).equals("-null-")) {
                if (!first) {
                    sb.append(", ");
                }
                sb.append(inst.attribute(i).name()).append("=").append(attributes.get(i));
                first = false;
            }
        }
    }
    
    /**
     * Méthode pour extraire des règles basées sur un critère
     */
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
     * Nouvelle méthode pour extraire des règles basées sur une connexion de Galois
     */
    public ArrayList<Classification_Rule> ExtraireRegleGalois(Instances inst, int conceptStrategy, int galoisStrategy) throws Exception {
        
        if(m_Debug) {
            printDebugHeader(inst);
            System.out.println("\nExtracting rules with Concept Strategy: " + conceptStrategy + 
                               " and Galois Strategy: " + galoisStrategy);
        }
        
        // Initialiser les évaluateurs d'attributs
        initializeFeatureEvaluator(inst);
        
        // Appliquer la stratégie Galois pour extraire les règles
        switch (galoisStrategy) {
            case GALOIS_PERTINENT_MAJORITY:
                if (conceptStrategy == CONCEPT_LEARNING_PERTINENT) {
                    return ExtraireRegleFermNom(inst, 1);
                } else {
                    return extractAllAttributesMajorityValue(inst);
                }
                
            case GALOIS_PERTINENT_ALL:
                if (conceptStrategy == CONCEPT_LEARNING_PERTINENT) {
                    return ExtraireRegleFermNom(inst, 3);
                } else {
                    return extractAllAttributesAllValues(inst);
                }
                
            case GALOIS_ALL_MAJORITY:
                if (conceptStrategy == CONCEPT_LEARNING_PERTINENT) {
                    return ExtraireRegleFermNom(inst, 2);
                } else {
                    return extractAllAttributesClassBased(inst);
                }
                
            case GALOIS_ALL_ALL:
                if (conceptStrategy == CONCEPT_LEARNING_PERTINENT) {
                    return extractAllAttributesAllValues(inst);
                } else {
                    return extractCompleteGalois(inst);
                }
                
            default:
                throw new IllegalArgumentException("Stratégie Galois non reconnue: " + galoisStrategy);
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
    private double[] computeFeatureMeasures(Instances inst) throws Exception {
        double[] featureMeasures = new double[inst.numAttributes()];
        Enumeration<Attribute> attEnum = inst.enumerateAttributes();
        
        while (attEnum.hasMoreElements()) {
            Attribute att = attEnum.nextElement();
            switch (FeatureMeasureMethod) {
                case FEAT_INFOGAIN:
                    featureMeasures[att.index()] = m_InfoGainAttributeEval.evaluateAttribute(att.index());
                    break;
                case FEAT_GAINRATIO:
                    featureMeasures[att.index()] = m_GainRatioAttributeEval.evaluateAttribute(att.index());
                    break;
                case FEAT_ENTROPY:
                    featureMeasures[att.index()] = computeEntropy(inst, att);
                    break;
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
     * Extrait des règles en utilisant tous les attributs mais seulement la valeur majoritaire
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
            
            // Trouver la valeur majoritaire
            int[] valueCounts = new int[attribute.numValues()];
            for (int i = 0; i < inst.numInstances(); i++) {
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
            
            if (m_Debug) {
                System.out.println("Majority value: " + attribute.value(majorityValueIdx) + 
                                 " with count: " + valueCounts[majorityValueIdx]);
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
            
            rules.add(rule);
        }
        
        return rules;
    }
    
    /**
     * Extrait des règles en utilisant tous les attributs et toutes leurs valeurs
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
                
                rules.add(rule);
            }
        }
        
        return rules;
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
            if (classifier.classify_Instance_nom(inst.instance(i), rule) != -1.0) {
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
        System.out.println("\n--- CALCUL DE LA FERMETURE DE GALOIS ---\n");
        System.out.println("Liste des attributs d'entrée:\n");
        printDatasetTable(inst);
        System.out.println("\n\n");
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
            System.out.println("Génération du CNC en cours ...");

        // Trouver la meilleure valeur distincte selon le support
        int supportDistVal = 0;
        int indexBestDistVal = 0;
        int suppBestDistVal = 0;

        // Pour traquer les ex aequo
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
                System.out.println("- Valeur \"" + currentVal + "\" a un support de : " + supportDistVal);
            }

            if (supportDistVal > suppBestDistVal) {
                suppBestDistVal = supportDistVal;
                indexBestDistVal = i;
                bestIndexes.clear();
                bestIndexes.add(i);
            } else if (supportDistVal == suppBestDistVal) {
                bestIndexes.add(i); // match ex aequo
                if (Policy_Selection == RuleExtractor.POLICY_LAST) {
                    indexBestDistVal = i; // on garde le dernier
                } else if (Policy_Selection == RuleExtractor.POLICY_FIRST) {
                    // on garde le premier déjà sélectionné
                }
            }
        }

        // Affichage final de la meilleure valeur
        if (m_Debug) {
            System.out.println();

            if (bestIndexes.size() > 1) {
                System.out.print("Attention : Plusieurs valeurs avec support maximal (" + suppBestDistVal + ") : ");
                for (int idx : bestIndexes) {
                    System.out.print("\"" + bestAttribute.value(idx) + "\" ");
                }
                System.out.println();

                if (Policy_Selection == RuleExtractor.POLICY_LAST) {
                    System.out.println("Politique de sélection : LAST → on sélectionne la derniere : \"" + bestAttribute.value(indexBestDistVal) + "\"\n");
                } else if (Policy_Selection == RuleExtractor.POLICY_FIRST) {
                    System.out.println("Politique de sélection : FIRST → on sélectionne la premiere : \"" + bestAttribute.value(bestIndexes.get(0)) + "\"\n");
                }
            } else {
                System.out.println("Meilleure valeur unique : \"" + bestAttribute.value(indexBestDistVal) + "\" avec un support de : " + suppBestDistVal + "\n");
            }

            System.out.println("L'attribut retenu pour calculer la fermeture est : " + bestAttribute.name() + 
                               " (index: " + bestAttribute.index() + ")");

        }
        
        // Trouver les instances qui ont cette valeur d'attribut
        ArrayList<Integer> instances = new ArrayList<>();
        for(int i=0; i<inst.numInstances(); i++) {
            if(inst.instance(i).stringValue(bestAttribute.index()).equals(bestAttribute.value(indexBestDistVal))) {
                instances.add(i);
            }
        }
        int numInstances = instances.size();
        
        // Calculer les attributs de la fermeture
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
        
        // Afficher la fermeture de Galois en utilisant la référence à RuleExtractor
        if (m_Debug && ruleExtractor != null) {
            ruleExtractor.printGaloisClosure(instances, attributes, inst);
        }
        
        if(m_Debug) {
            System.out.print("\nListe des valeurs d'attribut retenues:  ");
            for(int i=0; i<attributes.size()-1; i++)
            {
                if(!attributes.get(i).equals(nullValue)) {
                    System.out.print(attributes.get(i) + ", ");
                }
            }
        }
        
        // Trouver la classe majoritaire
        int majorityClassIndex = findMajorityClass(inst, instances);
        
        if(m_Debug)
            System.out.println("\n\nLa classe Majoritaire est: " + inst.attribute(inst.classIndex()).value(majorityClassIndex));
        
        // Créer la règle
        Classification_Rule rule = new Classification_Rule(
            attributes,
            majorityClassIndex,
            inst.attribute(inst.classIndex()).value(majorityClassIndex),
            featureMeasures[Utils.maxIndex(featureMeasures)]
        );
        
        // Ajout des noms des attributs à la règle
        ArrayList<String> attributeNames = new ArrayList<>();
        for (int i = 0; i < inst.numAttributes() - 1; i++) {
            attributeNames.add(inst.attribute(i).name());
        }
        rule.setAttributeNames(attributeNames);
        
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
            System.out.println("\nGénération d'un CNC à partir de la fermeture basée sur les classes pour l'attribut retenu");
        
        // Pour chaque classe
        for (int classIdx = 0; classIdx < inst.numClasses(); classIdx++) {
            if (m_Debug) {
                System.out.println("\nTraitement de la classe: " + inst.attribute(inst.classIndex()).value(classIdx));
            }
            
            // Collecter les instances de cette classe
            ArrayList<Integer> classInstances = new ArrayList<>();
            for (int i = 0; i < inst.numInstances(); i++) {
                if (inst.instance(i).classValue() == classIdx) {
                    classInstances.add(i);
                }
            }
            
            if (classInstances.isEmpty()) continue;
            
            // Trouver la valeur la plus fréquente pour cet attribut dans cette classe
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
            
            // Collecter les instances de cette classe avec cette valeur d'attribut
            ArrayList<Integer> relevantInstances = new ArrayList<>();
            for (int i : classInstances) {
                if (inst.instance(i).value(bestAttribute.index()) == majorityValueIdx) {
                    relevantInstances.add(i);
                }
            }
            
            if (relevantInstances.isEmpty()) continue;
            
            // Calculer la fermeture des attributs
            ArrayList<String> attributes = computeAttributeClosure(inst, relevantInstances);
            
            // Afficher la fermeture de Galois en utilisant la référence à RuleExtractor
            if (m_Debug && ruleExtractor != null) {
                ruleExtractor.printGaloisClosure(relevantInstances, attributes, inst);
            }
            
            // Créer la règle
            Classification_Rule rule = new Classification_Rule(
                attributes,
                classIdx,
                inst.attribute(inst.classIndex()).value(classIdx),
                featureMeasures[Utils.maxIndex(featureMeasures)]
            );
            
            // Ajout des noms des attributs à la règle
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
            System.out.println("\nGénération d'un CNC à partir de la fermeture de toutes les valeurs de l'attribut retenu");
        
        // Pour chaque valeur de l'attribut
        for (int valueIdx = 0; valueIdx < bestAttribute.numValues(); valueIdx++) {
            if (m_Debug) {
                System.out.println("\nTraitement de la valeur: " + bestAttribute.value(valueIdx));
            }
            
            // Trouver les instances avec cette valeur
            ArrayList<Integer> instances = new ArrayList<>();
            for (int i = 0; i < inst.numInstances(); i++) {
                if (inst.instance(i).value(bestAttribute.index()) == valueIdx) {
                    instances.add(i);
                }
            }
            
            if (instances.isEmpty()) continue;
            
            // Calculer la fermeture des attributs
            ArrayList<String> attributes = computeAttributeClosure(inst, instances);
            
            // Afficher la fermeture de Galois en utilisant la référence à RuleExtractor
            if (m_Debug && ruleExtractor != null) {
                ruleExtractor.printGaloisClosure(instances, attributes, inst);
            }
            
            // Trouver la classe majoritaire
            int majorityClassIndex = findMajorityClass(inst, instances);
            
            // Créer la règle
            Classification_Rule rule = new Classification_Rule(
                attributes,
                majorityClassIndex,
                inst.attribute(inst.classIndex()).value(majorityClassIndex),
                featureMeasures[Utils.maxIndex(featureMeasures)]
            );
            
            // Ajout des noms des attributs à la règle
            ArrayList<String> attributeNames = new ArrayList<>();
            for (int i = 0; i < inst.numAttributes() - 1; i++) {
                attributeNames.add(inst.attribute(i).name());
            }
            rule.setAttributeNames(attributeNames);
            
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