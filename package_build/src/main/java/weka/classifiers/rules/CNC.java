package weka.classifiers.rules;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Enumeration;
import java.util.Vector;

import weka.attributeSelection.*;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.classifiers.Sourcable;
import weka.core.*;
import weka.core.Capabilities.Capability;
import weka.core.Debug.Random;
import weka.classifiers.UpdateableClassifier;
import weka.classifiers.fca.Classification_Rule;
import weka.classifiers.fca.Classify_Instance;
import weka.classifiers.fca._GaloisConnection;
import weka.classifiers.fca.RuleExtractor;
import weka.core.TechnicalInformation.Field;
import weka.core.TechnicalInformation.Type;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.ReplaceMissingValues;

import java.text.SimpleDateFormat;
import java.util.Calendar;
import java.util.Collections;
import java.util.Date;

import weka.classifiers.Evaluation;

/**
 * CNC implementation that extracts classification rules
 * using concept learning and feature measures.
 */
public class CNC extends AbstractClassifier 
    implements UpdateableClassifier, TechnicalInformationHandler, WeightedInstancesHandler, Sourcable {

    static final long serialVersionUID = -2011091561919L;
    
    Calendar calendar;
    SimpleDateFormat sdf = new SimpleDateFormat("E MM/dd/yyyy HH:mm:ss.SSS");
    
    protected int m_CNC = 0;
    
    protected ReplaceMissingValues m_Missing = new ReplaceMissingValues();
    
    protected Classifier m_ZeroR;
    
    protected double m_LowPondVoter = (double) 66.66;
    
    public static ArrayList<Classification_Rule> m_classifierNC;
    
    protected static InfoGainAttributeEval m_InfoGainAttributeEval = new InfoGainAttributeEval();
    protected static GainRatioAttributeEval m_GainRatioAttributeEval = new GainRatioAttributeEval();

    // FeatureMeasureMethod is the method used to compute the feature measure
    
    public static final int FEAT_ENTROPY = 1;
    public static final int FEAT_INFOGAIN = 2;
    public static final int FEAT_GAINRATIO = 3;
    
    
    

    // seuil a dépasser
    protected double m_THRESHOLD = 0; 
   // Getter et Setter pour N
    public double getThreshold() {
        return m_THRESHOLD;
    }

    public void setThreshold(double s) {
        if (s > 0) {
            m_THRESHOLD = s;
        }
    }
    
    private int FeatureMeasureMethod = FEAT_INFOGAIN; // Default value : InfoGain
    
    public static final Tag[] TAGS_FeatureMeasure = {
        new Tag(FEAT_ENTROPY, "Entropy"),
        new Tag(FEAT_INFOGAIN, "Information Gain"),
        new Tag(FEAT_GAINRATIO, "Gain Ratio"),
    };
    
    // Each parameter is using two functions : a getter and a setter, in order to work on the code 
    public SelectedTag getFeatureMeasure() {
        return new SelectedTag(FeatureMeasureMethod, TAGS_FeatureMeasure);
    }
    
    public void setFeatureMeasure(SelectedTag value) {
        if (value.getTags() == TAGS_FeatureMeasure) {
            this.FeatureMeasureMethod = value.getSelectedTag().getID();
        }
    }

    
 // Types de classification nominale
    //public static final int AGGREGATION_SIMPLE = 1;  
    public static final int AGGREGATION_WEIGHTED_VOTE = 1;  
    public static final int AGGREGATION_MAJORITY_VOTE = 2;  
    public static final int AGGREGATION_PLURALITY_VOTE = 3;  
    public static final int AGGREGATION_MIN_PLURALITY_THRESHOLD = 4;  
    public static final int AGGREGATION_LOWEST_WEIGHT_VOTER =5 ;  
    public static final int AGGREGATION_TOP_K_VOTERS = 6;  

    private int AggregationStrategy = AGGREGATION_WEIGHTED_VOTE ; // Default value : vote ponderé
    
    
    public static final Tag[] TAGS_AGGREGATION = {
    	    new Tag(AGGREGATION_WEIGHTED_VOTE, "Weighted Vote Aggregation"),
    	    new Tag(AGGREGATION_MAJORITY_VOTE, "Majority Vote Aggregation"),
    	    new Tag(AGGREGATION_PLURALITY_VOTE, "Plurality Vote Aggregation"),
    	    new Tag(AGGREGATION_MIN_PLURALITY_THRESHOLD, "Plurality Vote Aggregation With Minimum Threshold"),
    	    new Tag(AGGREGATION_LOWEST_WEIGHT_VOTER, "Weighted Vote Aggregation With Minimum Weight Threshold"),
    	    new Tag(AGGREGATION_TOP_K_VOTERS, "Weighted Vote Aggregation With with Top-K Best Voters")
    	};
    
    public SelectedTag getAggregationStrategy() {
        return new SelectedTag(AggregationStrategy,TAGS_AGGREGATION);
    }

    public void setAggregationStrategy(SelectedTag value) {
        if (value.getTags() == TAGS_AGGREGATION) {
            this.AggregationStrategy = value.getSelectedTag().getID();
        }
    }
   
    
    public String getAggregationStrategyName() {
        switch (AggregationStrategy) {
           
            case AGGREGATION_WEIGHTED_VOTE:
                return "Weighted Vote Aggregation";
            case AGGREGATION_MAJORITY_VOTE:
                return "Majority Vote Aggregation";
            case AGGREGATION_PLURALITY_VOTE:
                return "Plurality Vote Aggregation";
            case AGGREGATION_MIN_PLURALITY_THRESHOLD:
                return "Plurality Vote Aggregation With Minimum Threshold";
            case AGGREGATION_LOWEST_WEIGHT_VOTER:
                return "Lowest Weight Voter Classification";
            case AGGREGATION_TOP_K_VOTERS:
                return "Top-K Best Voters Classification";
            default:
                return "Unknown Nominal Classification Strategy";
        }
    }
    
    
    
    
 // Déclarations des variables avec valeurs par défaut
    protected int m_TopKVoters = 10;
    protected double m_PluralityThreshold = 50;
    protected double m_WeightThreshold =10;

    // Getters et Setters pour TopKVoters
    /**
     * Retourne le nombre de top voters à considérer
     * @return le nombre de top voters
     */
    public int getTopKVoters() {
        return m_TopKVoters;
    }

    /**
     * Définit le nombre de top voters à considérer
     * @param numbTopKVotersVoters le nombre de top voters (doit être ≥ 1)
     * @throws IllegalArgumentException si la valeur est inférieure à 1
     */
    public void setTopKVoters(int numbTopKVotersVoters) {
        if (numbTopKVotersVoters < 1) {
            throw new IllegalArgumentException("Number of top voters must be at least 1");
        }
        m_TopKVoters = numbTopKVotersVoters;
    }

    // Getters et Setters pour PluralityThreshold
    /**
     * Retourne le seuil de pluralité minimum
     * @return le seuil de pluralité (entre 0.0 et 100)
     */
    public double getPluralityThreshold() {
        return m_PluralityThreshold;
    }

    /**
     * Définit le seuil de pluralité minimum
     * @param pluralityAgregationThreshold le seuil de pluralité (doit être entre 0.0 et 100)
     * @throws IllegalArgumentException si la valeur est hors limites
     */
    public void setPluralityThreshold(double pluralityAgregationThreshold) {
        if (pluralityAgregationThreshold < 0.0 || pluralityAgregationThreshold > 100) {
            throw new IllegalArgumentException("Plurality threshold must be between 0 and 100");
        }
        m_PluralityThreshold = pluralityAgregationThreshold;
    }

    // Getters et Setters pour WeightThreshold
    /**
     * Retourne le seuil de pondération minimum pour les voters
     * @return le seuil de pondération (entre 0.0 et 1.0)
     */
    public double getWeightThreshold() {
        return m_WeightThreshold;
    }

    /**
     * Définit le seuil de pondération minimum pour les voters
     * @param minWeightThreshold le seuil de pondération (doit être entre 0.0 et 1.0)
     * @throws IllegalArgumentException si la valeur est hors limites
     */
    public void setWeightThreshold(double minWeightThreshold) {
        if (minWeightThreshold < 0 || minWeightThreshold > 100) {
            throw new IllegalArgumentException("Weight threshold must be between 0.0 and 1.0");
        }
        m_WeightThreshold = minWeightThreshold;
    }
    
 // Constantes pour les stratégies
    public static final int CONCEPT_LEARNING_ALL_ALL = 1;
    public static final int CONCEPT_LEARNING_ALL_MAJORITY = 2;
    public static final int CONCEPT_LEARNING_PERTINENT_ALL = 3;
    public static final int CONCEPT_LEARNING_PERTINENT_MAJORITY = 4;
    public static final int CONCEPT_LEARNING_ABOVE_THRESHOLD_ALL = 5;
    public static final int CONCEPT_LEARNING_ABOVE_THRESHOLD_MAJORITY = 6;
    public static final int CONCEPT_LEARNING_UNDER_THRESHOLD_ALL = 7;
    public static final int CONCEPT_LEARNING_UNDER_THRESHOLD_MAJORITY = 8;

    
    private int ConceptLearningStrategy = CONCEPT_LEARNING_ALL_ALL; // Default value : All allF
    
    public void setConceptLearningStrategy(int strategy) {
        if (strategy >= CONCEPT_LEARNING_ALL_ALL && strategy <= CONCEPT_LEARNING_UNDER_THRESHOLD_MAJORITY) {
            this.ConceptLearningStrategy = strategy;
        } else {
            throw new IllegalArgumentException("Unknown Strategy : " + strategy);
        }
    }
    
    public String getConceptLearningStrategyName() {
        switch (ConceptLearningStrategy) {
            case CONCEPT_LEARNING_ALL_ALL: 
                return "Closure of All Attributes & All Values";
            case CONCEPT_LEARNING_ALL_MAJORITY: 
                return "Closure of All Attributes & Majority Value";
            case CONCEPT_LEARNING_PERTINENT_ALL: 
                return "Closure of Pertinent Attributes & All Values";
            case CONCEPT_LEARNING_PERTINENT_MAJORITY: 
                return "Closure of Pertinent Attributes & Majority Value";
            case CONCEPT_LEARNING_ABOVE_THRESHOLD_ALL: 
                return "Closure of Attributes Above Threshold & All Values";
            case CONCEPT_LEARNING_ABOVE_THRESHOLD_MAJORITY: 
                return "Closure of Attributes Above Threshold & Majority Value";
            case CONCEPT_LEARNING_UNDER_THRESHOLD_ALL: 
                return "Closure of Attributes Under Threshold & All Values";
            case CONCEPT_LEARNING_UNDER_THRESHOLD_MAJORITY: 
                return "Closure of Attributes Under Threshold & Majority Value";
            default: 
                return "Unknown Strategy";
        } 
    }
  
    
    
    public static final Tag[] TAGS_ConceptLearning = {
        new Tag(CONCEPT_LEARNING_ALL_ALL , "Closure of All Attributes & All Values"),
        new Tag(CONCEPT_LEARNING_ALL_MAJORITY, "Closure of ALL attributes & Majority Value"),
        new Tag(CONCEPT_LEARNING_PERTINENT_ALL, "Closure of the Pertinent Attribute & All Values"),
        new Tag(CONCEPT_LEARNING_PERTINENT_MAJORITY, "Closure of the Pertinent attribute & Majority Value"),
        new Tag(CONCEPT_LEARNING_ABOVE_THRESHOLD_ALL, "Closure of attributes above THRESHOLD & All Values"),
        new Tag(CONCEPT_LEARNING_ABOVE_THRESHOLD_MAJORITY,"Closure of attributes above THRESHOLD & Majority Value"),
        new Tag(CONCEPT_LEARNING_UNDER_THRESHOLD_ALL, "Closure of attributes under THRESHOLD & All Values"),
        new Tag(CONCEPT_LEARNING_UNDER_THRESHOLD_MAJORITY, "Closure of attributes under THRESHOLD & Majority Value")
        };
   
    
    public SelectedTag getConceptLearning() {
        return new SelectedTag(ConceptLearningStrategy, TAGS_ConceptLearning);
    }
    
    public void setConceptLearning(SelectedTag value) {
        if (value.getTags() == TAGS_ConceptLearning) {
            this.ConceptLearningStrategy = value.getSelectedTag().getID();
        }
    }

   

    // Policy_Selection is the policy used to select the attribute in case of equality
    
    public static final int POLICY_FIRST = 1;
    public static final int POLICY_LAST = 2;
    
    private int Policy_Selection = POLICY_LAST; // Default value : Last
    
    public static final Tag[] TAGS_Policy_Selection = {
        new Tag(POLICY_FIRST, "First occurence in case of equality"),
        new Tag(POLICY_LAST, "Last occurence in case of equality"),
    };
    
    public SelectedTag getPolicy_Selection() {
        return new SelectedTag(Policy_Selection, TAGS_Policy_Selection);
    }
    
    public void setPolicy_Selection(SelectedTag value) {
        if (value.getTags() == TAGS_Policy_Selection) {
            this.Policy_Selection = value.getSelectedTag().getID();
        }
    }

    // Filter is the filter used to preprocess the data before classification
    
    protected static Filter m_Filter = new weka.filters.supervised.attribute.Discretize();
    
    public void setFilter(Filter filter) {
        m_Filter = filter;
    }
    
    public Filter getFilter() {
        return m_Filter;
    }
    
    protected String getFilterSpec() {
        Filter c = getFilter();
        if (c instanceof OptionHandler) {
            return c.getClass().getName() + " "
                    + Utils.joinOptions(((OptionHandler)c).getOptions());
        }
        return c.getClass().getName();
    }
    
    protected Instances m_FilteredInstances;

    
    @Override
    public TechnicalInformation getTechnicalInformation() {
        TechnicalInformation result;
        
        result = new TechnicalInformation(Type.INPROCEEDINGS);
        result.setValue(Field.YEAR, "2011");
        result.setValue(Field.AUTHOR, "N. MEDDOURI");
        result.setValue(Field.PUBLISHER, "Academic Publisher");
        result.setValue(Field.ADDRESS, "Tunis, Tunisia");
        result.setValue(Field.ORGANIZATION, "Research Unit on Programming, Algorithmics and Heuristics");
        result.setValue(Field.SCHOOL, "Faculty of Science");
        
        return result;
    }
    
    public String globalInfo() {
        return "Classifier Nominal Concepts (CNC) discovers classification Rules using concept learning "
             + "and feature measures.\n\n"
             + "For more information, see:\n\n"
             + getTechnicalInformation().toString() + "\n\nThis code was rewritten and improved by Francoise Lapetite & Younes Kara Zaitri \n";
    }
    
    public Capabilities getCapabilities() {
        Capabilities result = super.getCapabilities();
        result.disableAll();
        
        result.enable(Capability.BINARY_ATTRIBUTES);
        result.enable(Capability.NOMINAL_ATTRIBUTES);
        result.enable(Capability.NUMERIC_ATTRIBUTES);
        
        result.enable(Capability.MISSING_VALUES);
        
        result.disableAllClasses();
        result.disableAllClassDependencies();
        result.enable(Capability.NOMINAL_CLASS);
        result.enable(Capability.BINARY_CLASS);
        result.enable(Capability.MISSING_CLASS_VALUES);
        
        result.setMinimumNumberInstances(1);
        
        return result;
    }
    
 // listOptions() returns a list of the options available for this classifier
   
    public Enumeration<Option> listOptions() {
        Vector<Option> newVector = new Vector<>(16);  

        
        newVector.addElement(new Option( 
        	    "\tCustom classifier key for display in Weka (not used internally).",
        	    "CNC_ID", 1, "-CNC_ID <string>"));

        // Options pour la stratégie de classification nominale
        newVector.addElement(new Option(
                "\t1: Weighted Vote Aggregation\n" +
                "\t2: Majority Vote Aggregation\n" +
                "\t3: Plurality Vote Aggregation\n" +
                "\t4: Minimum Plurality Threshold\n" +
                "\t5: Lowest Weight Voter\n" +
                "\t6: Top-K Best Voters",
                "aggregationStrategy", 1, "-aggregationStrategy <num>"));

        // Option pour le nombre K des meilleurs voters
        newVector.addElement(new Option(
                "\tNumber of top voters to consider (for Top-K strategy)",
                "numbTopKVoters", 1, "-numbTopKVoters <num>"));

        // Option pour le seuil de pluralité
        newVector.addElement(new Option(
                "\tMinimum plurality threshold value (0.0-100)",
                "pluralityAgregationThreshold", 1, "-pluralityAgregationThreshold <num>"));

        // Nouvelle option pour le seuil de pondération
        newVector.addElement(new Option(
                "\tWeight threshold for lowest weight voter (0.0-100)",
                "minWeightThreshold", 1, "-minWeightThreshold <num>"));

        // Option pour la stratégie de concept learning
        newVector.addElement(new Option(
                "\tConcept learning strategy:\n" +
                "\t1: All Attributes - All Values\n" +
                "\t2: All Attributes - Majority Value\n" +
                "\t3: Pertinent Attributes - All Values\n" +
                "\t4: Pertinent Attributes - Majority Value\n" +
                "\t5: Above Threshold - All Values\n" +
                "\t6: Above Threshold - Majority Value\n" +
                "\t7: Under Threshold - All Values\n" +
                "\t8: Under Threshold - Majority Value",
                "conceptStrategy", 1, "-conceptStrategy <num>"));

        // Option pour le seuil
        newVector.addElement(new Option(
            "\t Threshold value for attribute selection (>= 0 or -1 for automatic average).",
            
            "THRESHOLD", 1, "-THRESHOLD <num>"));

        // Option pour la mesure de caractéristique
        newVector.addElement(new Option(
            "\tFeature selection measure:",
            "FeatureMeasure", 1, "-FeatureMeasure <entropy|infogain|gainratio>"));

        // Option pour le filtre
        newVector.addElement(new Option(
            "\tFull filter specification:\n" +
            "\teg: \"weka.filters.supervised.attribute.Discretize -R first-last\"",
            "F", 1, "-F <filter specification>"));

        // Option pour le debug
        newVector.addElement(new Option(
            "\tEnable debug output",
            "debug", 0, "-debug"));

        // Options de la classe parente
        newVector.addAll(Collections.list(super.listOptions()));

        return newVector.elements();
    }
    
    
    public void setOptions(String[] options) throws Exception {
        // Reset des valeurs par défaut
        ConceptLearningStrategy = CONCEPT_LEARNING_ALL_ALL; // Valeur par défaut
        AggregationStrategy = AGGREGATION_WEIGHTED_VOTE; // Valeur par défaut
     // Option factice pour affichage uniquement
        Utils.getOption("CNC_ID", options); // Ne rien en faire, juste pour vider les options restantes

        // Gestion de la stratégie de classification nominale
        String aggregationStrategyStr = Utils.getOption("aggregationStrategy", options);
        if (!aggregationStrategyStr.isEmpty()) {
            int strategy = Integer.parseInt(aggregationStrategyStr);
            if (strategy < 1 || strategy > 6) {
                throw new IllegalArgumentException("Invalid aggregation strategy number. Must be between 1-6");
            }
            AggregationStrategy = strategy;
        }

        // Gestion du nombre de top voters
        String numbTopKVotersStr = Utils.getOption("numbTopKVoters", options);
        if (!numbTopKVotersStr.isEmpty()) {
            m_TopKVoters = Integer.parseInt(numbTopKVotersStr);
            if (m_TopKVoters < 1) {
                throw new IllegalArgumentException("Number of top voters must be at least 1");
            }
        }

        // Gestion du seuil de pluralité
        String pluralityAgregationThresholdStr = Utils.getOption("pluralityAgregationThreshold", options);
        if (!pluralityAgregationThresholdStr.isEmpty()) {
            m_PluralityThreshold = Double.parseDouble(pluralityAgregationThresholdStr);
            if (m_PluralityThreshold < 0 || m_PluralityThreshold > 100) {
                throw new IllegalArgumentException("Plurality threshold must be between 0 and 100");
            }
        }

        // Gestion du seuil de pondération
        String minWeightThresholdStr = Utils.getOption("minWeightThreshold", options);
        if (!minWeightThresholdStr.isEmpty()) {
            m_WeightThreshold= Double.parseDouble(minWeightThresholdStr);
            if (m_WeightThreshold < 0 || m_WeightThreshold > 100) {
                throw new IllegalArgumentException("Weight threshold must be between 0 and 100");
            }

        }

        // Gestion de la stratégie de concept learning (existante)
        String strategyStr = Utils.getOption("conceptStrategy", options);
        if (!strategyStr.isEmpty()) {
            int strategy = Integer.parseInt(strategyStr);
            if (strategy < 1 || strategy > 8) {
                throw new IllegalArgumentException("Invalid concept strategy number. Must be between 1-8");
            }
            ConceptLearningStrategy = strategy;
        }

     // Gestion du seuil (avec autorisation de -1 pour moyenne automatique)
        String thresholdStr = Utils.getOption("THRESHOLD", options);
        if (!thresholdStr.isEmpty()) {
            double parsedThreshold = Double.parseDouble(thresholdStr);
            if (parsedThreshold < -1.0) {
                throw new IllegalArgumentException("Threshold must be -1 (auto) or >= 0.");
            }
            setThreshold(parsedThreshold);
        }

        // Gestion de la mesure de caractéristique (existante)
        String measureStr = Utils.getOption("FeatureMeasure", options);
        if (!measureStr.isEmpty()) {
            switch (measureStr.toLowerCase()) {
                case "entropy":
                    FeatureMeasureMethod = FEAT_ENTROPY; 
                    break;
                case "infogain":
                    FeatureMeasureMethod = FEAT_INFOGAIN;
                    break;
                case "gainratio":
                    FeatureMeasureMethod = FEAT_GAINRATIO;
                    break;
                default:
                    throw new IllegalArgumentException("Invalid feature measure: " + measureStr);
            }
        }

        // Gestion du filtre (existante)
        String filterStr = Utils.getOption('F', options);
        if (!filterStr.isEmpty()) {
            String[] filterSpec = Utils.splitOptions(filterStr);
            if (filterSpec.length == 0) {
                setFilter(new weka.filters.supervised.attribute.Discretize());
            } else {
                try {
                    String filterName = filterSpec[0];
                    filterSpec[0] = "";
                    Filter filter = (Filter) Utils.forName(Filter.class, filterName, filterSpec);
                    setFilter(filter);
                } catch (Exception e) {
                    System.err.println("Warning: Could not set filter, using default Discretize. Error: " + e.getMessage());
                    setFilter(new weka.filters.supervised.attribute.Discretize());
                }
            }
        } else {
            setFilter(new weka.filters.supervised.attribute.Discretize());
        }

        // Gestion du debug (existante)
        m_Debug = Utils.getFlag("debug", options);

        super.setOptions(options);
    }

    
    public String getIdentifier() {
        StringBuilder id = new StringBuilder();

        id.append("CNC");
        id.append("_Agg=").append(getAggregationStrategyName());

        if (AggregationStrategy == AGGREGATION_TOP_K_VOTERS) {
            id.append("_TopK=").append(m_TopKVoters);
        }

        if (AggregationStrategy == AGGREGATION_MIN_PLURALITY_THRESHOLD) {
            id.append("_PluralityThr=").append(m_PluralityThreshold);
        }

        if (AggregationStrategy == AGGREGATION_LOWEST_WEIGHT_VOTER) {
            id.append("_MinWeightThr=").append(m_WeightThreshold);
        }

        id.append("_ConceptLearning=").append(getConceptLearningStrategyName());

        if (ConceptLearningStrategy >= CONCEPT_LEARNING_ABOVE_THRESHOLD_ALL &&
            ConceptLearningStrategy <= CONCEPT_LEARNING_UNDER_THRESHOLD_MAJORITY) {
            id.append("_Threshold=").append(m_THRESHOLD);
        }

        id.append("_Measure=").append(featureMeasureToString(FeatureMeasureMethod));

        return id.toString();
    }


    
    public String[] getOptions() {
        Vector<String> result = new Vector<>();
        
     // Clé personnalisée pour Weka Experimenter
        result.add("-CNC_ID");
        result.add(getIdentifier());
        
        

     // strategies d aggregation
        result.add("-aggregationStrategy");
        result.add(String.valueOf(AggregationStrategy));


        // Nombre de top voters
        result.add("-numbTopKVoters");
        result.add(String.valueOf(m_TopKVoters));

        // Seuil de pluralité
        result.add("-pluralityAgregationThreshold");
        result.add(String.valueOf(m_PluralityThreshold));

        // Seuil de pondération
        result.add("-minWeightThreshold");
        result.add(String.valueOf(m_WeightThreshold));

        // Stratégie de concept learning 
        result.add("-conceptStrategy");
        result.add(String.valueOf(ConceptLearningStrategy));

        // Seuil 
        result.add("-THRESHOLD");
        result.add(String.valueOf(getThreshold()));

        // Mesure de caractéristique 
        result.add("-FeatureMeasure");
        switch (FeatureMeasureMethod) {
            case FEAT_ENTROPY:
                result.add("entropy");
                break;
            case FEAT_INFOGAIN:
                result.add("infogain");
                break;
            case FEAT_GAINRATIO:
                result.add("gainratio");
                break;
        }

        // Filtre (existante)
        result.add("-F");
        result.add(getFilterSpec());

        // Debug (existante)
        if (m_Debug) {
            result.add("-debug");
        }

        // Options de la classe parente
        Collections.addAll(result, super.getOptions());

        return result.toArray(new String[0]);
    }
    
    
    
    // classifyInstance() classifies a given instance, it is mandatory to implement this function
    @Override
    public double classifyInstance(Instance inst) throws Exception {
        // Vérification préalable
        if (m_classifierNC == null) {
            throw new IllegalStateException("Model not built. Call buildClassifier() first.");
        }

        // Filtrage unique
        m_Filter.input(inst);
        m_Filter.batchFinished();
        Instance filteredInst = m_Filter.output();
        Classify_Instance classifier = new Classify_Instance();
 
        double result =-1.0;
        int nb_class = Classification_Rule.getMaxNumClass(m_classifierNC); // Assuming such a method exists
        switch (AggregationStrategy) {
       
        case AGGREGATION_WEIGHTED_VOTE :
            // Classification par vote pondéré
            result = classifier.classify_Instance_nom_VotePond(
                filteredInst, 
                m_classifierNC, 
                nb_class,
                m_Debug
            );

            break;
            
        case AGGREGATION_MAJORITY_VOTE:
            // Classification par vote majoritaire
            result = classifier.classify_Instance_nom_VoteMaj(
                filteredInst, 
                m_classifierNC, 
                nb_class,
                m_Debug
            );

            break;
            
        case AGGREGATION_PLURALITY_VOTE:
            // Classification par vote pluralitaire
            result = classifier.classify_Instance_nom_VotePlur(
                filteredInst, 
                m_classifierNC, 
                nb_class,
                m_Debug
            );

            break;
            
        case AGGREGATION_MIN_PLURALITY_THRESHOLD:
            // Classification avec seuil minimum de pluralité
            result = classifier.classify_Instance_nom_VotePlurMin(
                filteredInst, 
                m_classifierNC, 
                nb_class,
                m_PluralityThreshold,
                m_Debug
            );

            break;
            
        case AGGREGATION_LOWEST_WEIGHT_VOTER:

            // Classification en excluant les voters de plus faible poids
            result = classifier.classify_Instance_nom_VoteLowPondVoter(
                filteredInst, 
                m_classifierNC, 
                nb_class,
                m_WeightThreshold,
                m_Debug
            );


            break;
            
        case AGGREGATION_TOP_K_VOTERS:
            // Classification utilisant seulement les K meilleurs voters
            result = classifier.classify_Instance_nom_VoteNumBestVoterS(
                filteredInst, 
                m_classifierNC, 
                nb_class,
                m_TopKVoters,
                m_Debug
            );

            break;
            
        default:
            throw new Exception("Stratégie de classification nominale inconnue");
            
            
       
    }
        
       
        
        

        return result == -1.0 ? Utils.missingValue() : result;
    }

    

    
    
    // buildClassifier() builds the classifier using the training data, it is mandatory to implement this function

    
    public void buildClassifier(Instances data) throws Exception {
        // Vérification des capacités et prétraitement des données
        getCapabilities().testWithFail(data);
        
        // Étape 1: Préparation des données
        m_FilteredInstances = new Instances(data);
        m_FilteredInstances.deleteWithMissingClass();
        
        if (m_FilteredInstances.numInstances() == 0)
            throw new Exception("No training instances left after removing instances with MissingClass!");
        
        // Gestion des valeurs manquantes
        m_Missing.setInputFormat(m_FilteredInstances);
        m_FilteredInstances = Filter.useFilter(m_FilteredInstances, m_Missing);
        
        if (m_FilteredInstances.numInstances() == 0)
            throw new Exception("No training instances left after removing instances with MissingValues!");
        
        // Application du filtre (discretisation)
        m_Filter.setInputFormat(m_FilteredInstances);
        m_FilteredInstances = Filter.useFilter(m_FilteredInstances, m_Filter);
        
        // Cas particulier: seulement l'attribut classe
        if (m_FilteredInstances.numAttributes() == 1) {
            System.err.println("Cannot build model (only class attribute present in data!), "
                + "using ZeroR model instead!");
            m_ZeroR = new weka.classifiers.rules.ZeroR();
            m_ZeroR.buildClassifier(m_FilteredInstances);
            return;
        } else {
            m_ZeroR = null;
            this.m_CNC = 1;
        }
        
        // Étape 2: Sélection de la stratégie de génération de règles
     
        RuleExtractor extractor = new RuleExtractor(m_Debug, FeatureMeasureMethod, Policy_Selection, ConceptLearningStrategy);

        switch (ConceptLearningStrategy) {
            case CONCEPT_LEARNING_ALL_ALL:
                if (m_Debug) {
                    System.out.println("=== ALL ATTRIBUTES - ALL VALUES MODE ACTIVE ===");
                    System.out.println("Extraction using all attributes with all values");
                    System.out.println("Used Measure: " + featureMeasureToString(FeatureMeasureMethod));
                }
                m_classifierNC = extractor.extractAllAttributesAllValues(m_FilteredInstances);
                if (m_Debug) {
                    System.out.println("All attributes used");
                    printRuleStatistics();
                }
                break;
                
            case CONCEPT_LEARNING_ALL_MAJORITY:
                if (m_Debug) {
                    System.out.println("=== ALL ATTRIBUTES - MAJORITY VALUE MODE ACTIVE ===");
                    System.out.println("Extraction using all attributes with majority values");
                    System.out.println("Used Measure: " + featureMeasureToString(FeatureMeasureMethod));
                }
                m_classifierNC = extractor.extractAllAttributesMajorityValue(m_FilteredInstances);
                if (m_Debug) {
                    printRuleStatistics();
                }
                break;
                
            case CONCEPT_LEARNING_PERTINENT_ALL:
                if (m_Debug) {
                    System.out.println("=== PERTINENT ATTRIBUTE - ALL VALUES MODE ACTIVE ==="); 
                    System.out.println("Extraction using only pertinent attribute with all values");
                    System.out.println("Used Measure: " + featureMeasureToString(FeatureMeasureMethod));
                    extractor.printPertinentAttribute(m_FilteredInstances);

                }
                m_classifierNC = extractor.extractPertinentAttributeAllValues(m_FilteredInstances);
                if (m_Debug) {
                    printRuleStatistics();
                }

                break;
                
            case CONCEPT_LEARNING_PERTINENT_MAJORITY:
                if (m_Debug) {
                    System.out.println("=== PERTINENT ATTRIBUTES - MAJORITY VALUE MODE ACTIVE ===");
                    System.out.println("Extraction using only pertinent attribute with majority value");
                    System.out.println("Used Measure: " + featureMeasureToString(FeatureMeasureMethod));
                    extractor.printPertinentAttribute(m_FilteredInstances);

                } 
                m_classifierNC = extractor.extractPertinentAttributeMajorityValue(m_FilteredInstances); 
                if (m_Debug) {
                    printRuleStatistics();
                }

                break;
                
            case CONCEPT_LEARNING_ABOVE_THRESHOLD_ALL:
                if (m_Debug) {
                    System.out.println("=== ABOVE THRESHOLD - ALL VALUES MODE ACTIVE ===");
                    System.out.println("Extraction with THRESHOLD " + m_THRESHOLD);
                    System.out.println("Used Measure: " + featureMeasureToString(FeatureMeasureMethod));
                    extractor.printAttributesAboveThreshold(m_FilteredInstances, m_THRESHOLD);

                }
                m_classifierNC = extractor.extractAttributeAboveThresholdAllValues(m_FilteredInstances,m_THRESHOLD);
                if (m_Debug) {
                    printRuleStatistics();
                }

                break;
                
            case CONCEPT_LEARNING_ABOVE_THRESHOLD_MAJORITY:
                if (m_Debug) {
                    System.out.println("=== ABOVE THRESHOLD - MAJORITY VALUES MODE ACTIVE ===");
                    System.out.println("Extraction with THRESHOLD " + m_THRESHOLD);
                    System.out.println("Used Measure: " + featureMeasureToString(FeatureMeasureMethod));
                    extractor.printAttributesAboveThreshold(m_FilteredInstances, m_THRESHOLD);

                }
                m_classifierNC = extractor.extractAttributeAboveThresholdMajorityValues(m_FilteredInstances,m_THRESHOLD);
                if (m_Debug) {
                    printRuleStatistics();
                }

                break;
                
            case CONCEPT_LEARNING_UNDER_THRESHOLD_ALL:
                if (m_Debug) {
                    System.out.println("=== UNDER THRESHOLD - ALL VALUES MODE ACTIVE ===");
                    System.out.println("Extraction with THRESHOLD " + m_THRESHOLD);
                    System.out.println("Used Measure: " + featureMeasureToString(FeatureMeasureMethod));
                    extractor.printAttributesUnderThreshold(m_FilteredInstances, m_THRESHOLD);

                }
                m_classifierNC = extractor.extractAttributeUnderThresholdAllValues(m_FilteredInstances,m_THRESHOLD);
                if (m_Debug) {
                    printRuleStatistics();
                }

                break;
                
            case CONCEPT_LEARNING_UNDER_THRESHOLD_MAJORITY:
                if (m_Debug) {
                    System.out.println("=== UNDER THRESHOLD - MAJORITY VALUES MODE ACTIVE ===");
                    System.out.println("Extraction with THRESHOLD " + m_THRESHOLD);
                    System.out.println("Used Measure: " + featureMeasureToString(FeatureMeasureMethod));
                    extractor.printAttributesUnderThreshold(m_FilteredInstances, m_THRESHOLD);

                }
                m_classifierNC = extractor.extractAttributeUnderThresholdMajorityValues(m_FilteredInstances,m_THRESHOLD);
                if (m_Debug) {
                    printRuleStatistics();
                }

                break;
        }

        
    }
    
  
    
    public static String featureMeasureToString(int measureType) {
        switch (measureType) {
            case FEAT_ENTROPY:
                return "Entropy";
            case FEAT_INFOGAIN:
                return "Information Gain";
            case FEAT_GAINRATIO:
                return "Gain Ratio";
            default:
                return "Unknown Measure";
        }
    }

    // Méthode helper pour créer une règle à partir d'instances
    private Classification_Rule createRuleFromInstances(ArrayList<Integer> instanceIndices, int attrIndex, int valueIdx) {
        Instances instances = m_FilteredInstances;
        ArrayList<String> attributes = new ArrayList<>();
        for (int a = 0; a < instances.numAttributes() - 1; a++) {
            attributes.add("-null-");
        }
        
        int fermetureIdx = (attrIndex < instances.classIndex()) ? attrIndex : attrIndex - 1;
        attributes.set(fermetureIdx, instances.attribute(attrIndex).value(valueIdx));
        
        int[] classCounts = new int[instances.numClasses()];
        for (int idx : instanceIndices) {
            classCounts[(int)instances.instance(idx).classValue()]++;
        } 
        
        int majorityClass = findMajorityClass(classCounts);
        
        Classification_Rule rule = new Classification_Rule(
            attributes,
            majorityClass,
            instances.classAttribute().value(majorityClass)
        );
        
        rule.setRule_Ponderation(calculateRuleWeight(instanceIndices.size(), instances.numInstances()));
        
        return rule;
    }

    // Méthode helper pour trouver la classe majoritaire
    private int findMajorityClass(int[] classCounts) {
        int majorityClass = 0;
        for (int c = 0; c < classCounts.length; c++) {
            if (classCounts[c] > classCounts[majorityClass]) {
                majorityClass = c;
            }
        }
        return majorityClass;
    }

    // Méthode helper pour calculer le poids
    private double calculateRuleWeight(int coveredInstances, int totalInstances) {
        return (double)coveredInstances / totalInstances;
    }

    // Méthode helper pour l'affichage debug
    private void printRuleStatistics() {
        System.out.println("\n\n--- RULE GENERATION STATISTICS ---\n");
        System.out.println("Total rules generated: " + m_classifierNC.size());
        
        for (int i = 0; i < m_classifierNC.size(); i++) {
            Classification_Rule rule = m_classifierNC.get(i);
            
            System.out.printf("Rule #%d:\n", i + 1);
            System.out.println(rule.affich_nom_rule(true));
            
            System.out.println("----------------------------\n");
        }
        
    }

    
    
    
    
    
    
    
    
    private double evaluateErrorRate(Instances data) throws Exception {
    	double m_ErrorRate;
        Evaluation eval = new Evaluation(data);
        eval.evaluateModel(this, data);
        m_ErrorRate = eval.errorRate();
        
        // Vous pouvez aussi capturer d'autres métriques
        if (m_Debug) {
            System.out.println(eval.toSummaryString());
        }
        return m_ErrorRate;
    }
    
    

    private boolean isInExperimentMode() {
        // Vérifie la pile d'appels pour détecter le contexte d'expérimentation
        return Arrays.stream(Thread.currentThread().getStackTrace())
                .anyMatch(e -> e.getClassName().contains("Experiment"));
    }
    
    
    public String toString() {

    	// Vérifie si on est en mode expérimentation
        if (isInExperimentMode()) {
            return super.toString(); // ou return "";
        }
        
        StringBuilder sb = new StringBuilder();
        SimpleDateFormat sdf = new SimpleDateFormat("EEE yyyy-MM-dd HH:mm:ss.SSS");
        String currentTime = sdf.format(new Date());
        
        // En-tête du classifieur
        sb.append("=== CNC Classifier Output ===\n");
        sb.append("Generated at: ").append(currentTime).append("\n\n");
        
        // Configuration du classifieur
        sb.append("=== Configuration ===\n");
        sb.append(String.format("%-25s: %s\n", "Concept Learning Strategy", getConceptLearningStrategyName()));
        sb.append(String.format("%-25s: %s\n", "Feature Measure", featureMeasureToString(FeatureMeasureMethod)));
        sb.append(String.format("%-25s: %.4f\n", "THRESHOLD", m_THRESHOLD));  // Affichage du seuil avec 4 décimales
        sb.append(String.format("%-25s: %s\n", "Policy Selection", getPolicySelectionName()));
        sb.append(String.format("%-25s: %s\n", "Aggregation Strategy", getAggregationStrategyName()));
        sb.append(String.format("%-25s: %.4f\n", "Pularity Threshold", m_PluralityThreshold));  // Affichage du seuil avec 4 décimales
        sb.append(String.format("%-25s: %.4f\n", "Weight Threshold",m_WeightThreshold));  // Affichage du seuil avec 4 décimales
        sb.append(String.format("%-25s: %d\n", "TOP K Voters Threshold",m_TopKVoters));  

        
        

        sb.append("\n");
        
        // Statistiques de performance
        if (m_classifierNC != null && !m_classifierNC.isEmpty()) {
            sb.append("=== Performance ===\n");
           
            sb.append(String.format("%-25s: %d\n", "Number of Rules Generated", m_classifierNC.size()));
            sb.append("\n");
            
            // Détail des règles
            sb.append("=== Generated Rules ===\n");
            for (int i = 0; i < m_classifierNC.size(); i++) {
                sb.append(String.format("\nRule %d:\n", i + 1));
                sb.append(m_classifierNC.get(i).affich_nom_rule(true));
                sb.append("\n");
            }
        } else if (m_ZeroR != null) {
            sb.append("Using ZeroR model as fallback\n");
        } else {
            sb.append("No model generated\n");
        }
        
        // Information de débogage
        if (m_Debug) {
            sb.append("\n=== Debug Information ===\n");
            sb.append("Log console available\n");
            sb.append("Last execution: ").append(currentTime).append("\n");
        }
        double m_ErrorRate = -1;
        try {
            m_ErrorRate = evaluateErrorRate(this.m_FilteredInstances);
        } catch (Exception e) {
        	if (m_Debug) {
                System.out.println("Error in error rate calculation: " + e.getMessage());

        	}
        }
        if (m_ErrorRate >= 0) {
            sb.append(String.format("%-25s: %.2f%%\n", "Error Rate", m_ErrorRate * 100));
        }
        return sb.toString();
    }



    private String getPolicySelectionName() {
        switch (Policy_Selection) {
            case POLICY_FIRST: return "First Attribute";
            case POLICY_LAST: return "Last Attribute";
            default: return "Unknown";
        }
    }

    
    
    
    
    
    // Main function to run the classifier
    //
    public static void main(String[] args) {
    	runClassifier(new CNC(), args);
    	//test();
    	
    }
    	
    public static void test() {    	
        try {
            // Chemin vers un jeu de données inclus avec Weka si le vôtre ne fonctionne pas
        	String dataPath="C:\\Users\\asus\\Desktop\\weather.nominal.arff";
        	Instances data = DataSource.read(dataPath);
            data.setClassIndex(data.numAttributes() - 1);

            System.out.println("=== TEST AVEC DONNEES Weather ===");
            System.out.println("Nombre d'instances: " + data.numInstances());
            System.out.println("Nombre d'attributs: " + data.numAttributes());

            // Test avec différentes stratégies
            testStrategy(data, CNC.CONCEPT_LEARNING_PERTINENT_ALL, 0.2);
            

        } catch (Exception e) {
            System.err.println("Erreur lors du test: ");
            e.printStackTrace();
          
        }
    }
   
    private static void testStrategy(Instances data, int strategy, double THRESHOLD) throws Exception {
        // Initialisation du classifieur
        CNC classifier = new CNC();
        
        // Configuration
        classifier.setConceptLearningStrategy(strategy);
        if (strategy == CNC.CONCEPT_LEARNING_PERTINENT_MAJORITY) { 
            classifier.setThreshold(THRESHOLD);
        }
        classifier.setFeatureMeasure(new SelectedTag(FEAT_INFOGAIN, TAGS_FeatureMeasure));
        classifier.setDebug(true);  // Activer le mode debug
        
        // Construction du modèle
        System.out.println("\nConstruction du modèle...");
        classifier.buildClassifier(data);
        
        // Evaluation
        System.out.println("\nModel Evaluation:");
        Evaluation eval = new Evaluation(data);
        eval.crossValidateModel(classifier, data, 10, new Random(1));  // 10-fold CV
        
        // Affichage des résultats
        System.out.println("\n=== RESULTS ===");
        System.out.println("Strategy: " + classifier.getConceptLearningStrategyName() + 
                          (strategy == CNC.CONCEPT_LEARNING_PERTINENT_MAJORITY ? " (N=" + THRESHOLD + ")" : ""));
        System.out.println("Précision: " + eval.pctCorrect() + "%");
        System.out.println("Nombre de règles générées: " + CNC.m_classifierNC.size());
        //System.out.println("Temps de construction: " + eval.getCPUTime() + "s");
        
        // Affichage de quelques règles (debug)
        if (classifier.m_Debug && CNC.m_classifierNC.size() > 0) {
            System.out.println("\nExemple de règles générées:");
            for (int i = 0; i < Math.min(3, CNC.m_classifierNC.size()); i++) {
                System.out.println("Règle " + (i+1) + ": " + 
                    CNC.m_classifierNC.get(i).affich_nom_rule(true));
            }
        }
    }

    

	@Override
	public String toSource(String className) throws Exception {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public void updateClassifier(Instance instance) throws Exception {
		// TODO Auto-generated method stub
		
	}
}