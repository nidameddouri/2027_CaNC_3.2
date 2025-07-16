package weka.classifiers.rules;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Enumeration;
import java.util.List;
import java.util.Vector;
import java.text.SimpleDateFormat;
import java.util.Calendar;
import java.util.Collections;
import java.util.Date;

import weka.attributeSelection.*;
import weka.classifiers.*;
import weka.classifiers.fca.*;
import weka.classifiers.fca._GaloisConnection.FormalConcept;
import weka.core.*;
import weka.core.Capabilities.Capability;
import weka.core.TechnicalInformation.Field;
import weka.core.TechnicalInformation.Type;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.ReplaceMissingValues;

/**
 * CNC implementation that extracts classification rules
 * using concept learning and feature measures.
 */
public class CNC extends AbstractClassifier 
    implements UpdateableClassifier, TechnicalInformationHandler, WeightedInstancesHandler, Sourcable {

    // Constants and configuration
    static final long serialVersionUID = -2011091561919L;
    
    // Feature measure constants
    public static final int FEAT_ENTROPY = 1;
    public static final int FEAT_INFOGAIN = 2;
    public static final int FEAT_GAINRATIO = 3;
    
    //All Combinations constants
    
    // Threshold timing constants
    public static final int THRESHOLD_BEFORE_DISCRETIZATION = 1;
    public static final int THRESHOLD_AFTER_DISCRETIZATION = 2;
    
    // Aggregation strategy constants
    public static final int AGGREGATION_WEIGHTED_VOTE = 1;  
    public static final int AGGREGATION_MAJORITY_VOTE = 2;  
    public static final int AGGREGATION_PLURALITY_VOTE = 3;  
    public static final int AGGREGATION_MIN_PLURALITY_THRESHOLD = 4;  
    public static final int AGGREGATION_LOWEST_WEIGHT_VOTER = 5;  
    public static final int AGGREGATION_TOP_K_VOTERS = 6;  

    // Concept learning strategy constants
    public static final int CONCEPT_LEARNING_ALL_ALL = 1;
    public static final int CONCEPT_LEARNING_ALL_MAJORITY = 2;
    public static final int CONCEPT_LEARNING_PERTINENT_ALL = 3;
    public static final int CONCEPT_LEARNING_PERTINENT_MAJORITY = 4;
    public static final int CONCEPT_LEARNING_ABOVE_THRESHOLD_ALL = 5;
    public static final int CONCEPT_LEARNING_ABOVE_THRESHOLD_MAJORITY = 6;
    public static final int CONCEPT_LEARNING_UNDER_THRESHOLD_ALL = 7;
    public static final int CONCEPT_LEARNING_UNDER_THRESHOLD_MAJORITY = 8;
    public static final int N_ATTRIBUTES_COMBINATION_ALL_VALUES= 9;
    public static final int N_ATTRIBUTES_COMBINATION_MAJORITY_VALUES= 10;


    

    // Policy selection constants
    public static final int POLICY_FIRST = 1;
    public static final int POLICY_LAST = 2;
    
    private Instances structure;
    
    //All Combinations constants

    public static final boolean ALL_COMBINATIONS = true;

    //Concepts list
    String conceptsListStr;
    
    // Tag arrays for GUI options
    public static final Tag[] TAGS_THRESHOLD_TIMING = {
        new Tag(THRESHOLD_BEFORE_DISCRETIZATION, "Calculate threshold BEFORE discretization"),
        new Tag(THRESHOLD_AFTER_DISCRETIZATION, "Calculate threshold AFTER discretization")
    };
    
    public static final Tag[] TAGS_FeatureMeasure = {
        new Tag(FEAT_ENTROPY, "Entropy"),
        new Tag(FEAT_INFOGAIN, "Information Gain"),
        new Tag(FEAT_GAINRATIO, "Gain Ratio"),
    };
    
    public static final Tag[] TAGS_AGGREGATION = {
        new Tag(AGGREGATION_WEIGHTED_VOTE, "Weighted Vote Aggregation"),
        new Tag(AGGREGATION_MAJORITY_VOTE, "Majority Vote Aggregation"),
        new Tag(AGGREGATION_PLURALITY_VOTE, "Plurality Vote Aggregation"),
        new Tag(AGGREGATION_MIN_PLURALITY_THRESHOLD, "Plurality Vote Aggregation With Minimum Threshold"),
        new Tag(AGGREGATION_LOWEST_WEIGHT_VOTER, "Weighted Vote Aggregation With Minimum Weight Threshold"),
        new Tag(AGGREGATION_TOP_K_VOTERS, "Weighted Vote Aggregation With with Top-K Best Voters")
    };
    
    public static final Tag[] TAGS_ConceptLearning = {
        new Tag(CONCEPT_LEARNING_ALL_ALL, "Closure of All Attributes & All Values"),
        new Tag(CONCEPT_LEARNING_ALL_MAJORITY, "Closure of ALL attributes & Majority Value"),
        new Tag(CONCEPT_LEARNING_PERTINENT_ALL, "Closure of the Pertinent Attribute & All Values"),
        new Tag(CONCEPT_LEARNING_PERTINENT_MAJORITY, "Closure of the Pertinent attribute & Majority Value"),
        new Tag(CONCEPT_LEARNING_ABOVE_THRESHOLD_ALL, "Closure of attributes above THRESHOLD & All Values"),
        new Tag(CONCEPT_LEARNING_ABOVE_THRESHOLD_MAJORITY, "Closure of attributes above THRESHOLD & Majority Value"),
        new Tag(CONCEPT_LEARNING_UNDER_THRESHOLD_ALL, "Closure of attributes under THRESHOLD & All Values"),
        new Tag(CONCEPT_LEARNING_UNDER_THRESHOLD_MAJORITY, "Closure of attributes under THRESHOLD & Majority Value"),
        new Tag(N_ATTRIBUTES_COMBINATION_ALL_VALUES,"Closure of N attributes combination All Values" ),
        new Tag(N_ATTRIBUTES_COMBINATION_MAJORITY_VALUES,"Closure of N attributes combination Majority  Values" ),


    };
    
    public static final Tag[] TAGS_Policy_Selection = {
        new Tag(POLICY_FIRST, "First occurrence in case of equality"),
        new Tag(POLICY_LAST, "Last occurrence in case of equality"),
    };

    // Class members
    protected int m_CNC = 0;
    protected double m_LowPondVoter = 66.66;
    protected double m_THRESHOLD = 0;
    protected boolean Auto_THRESHOLD = true;
    protected int m_ThresholdTiming = THRESHOLD_BEFORE_DISCRETIZATION;
    protected int FeatureMeasureMethod = FEAT_GAINRATIO;
    protected int AggregationStrategy = AGGREGATION_WEIGHTED_VOTE;
    protected int ConceptLearningStrategy = CONCEPT_LEARNING_ALL_ALL;
    protected int Policy_Selection = POLICY_LAST;
    protected int m_TopKVoters = 10;
    protected double m_PluralityThreshold = 50;
    protected double m_WeightThreshold = 10;
    protected boolean allCombinations = true;
    protected int combinationSize = 1;


    
    protected ReplaceMissingValues m_Missing = new ReplaceMissingValues();
    protected Classifier m_ZeroR;
    protected static Filter m_Filter = new weka.filters.unsupervised.attribute.Discretize();
    protected Instances m_FilteredInstances;
    public static ArrayList<Classification_Rule> m_classifierNC;
    
    protected static InfoGainAttributeEval m_InfoGainAttributeEval = new InfoGainAttributeEval();
    protected static GainRatioAttributeEval m_GainRatioAttributeEval = new GainRatioAttributeEval();
    
    Calendar calendar;
    SimpleDateFormat sdf = new SimpleDateFormat("E MM/dd/yyyy HH:mm:ss.SSS");

    // Getters and Setters
    public double getThreshold() {
        return m_THRESHOLD;
    }

    public void setThreshold(double s) {
        if (s > 0) {
            m_THRESHOLD = s;
        }
    }
    
    public boolean getAuto_Threshold() {
        return Auto_THRESHOLD;
    }

    public void setAuto_Threshold(boolean s) {
        Auto_THRESHOLD = s;  
    }
    
    public String autoThresholdToString() {
        return Auto_THRESHOLD ? "Auto Generated" : "Manually Set";
    }
    
    public String m_ThresholdTimingToString() {
        return (m_ThresholdTiming == 1) ? "BEFORE discretization" : "AFTER discretization";
    }

    public SelectedTag getFeatureMeasure() {
        return new SelectedTag(FeatureMeasureMethod, TAGS_FeatureMeasure);
    }
    
    public void setFeatureMeasure(SelectedTag value) {
        if (value.getTags() == TAGS_FeatureMeasure) {
            this.FeatureMeasureMethod = value.getSelectedTag().getID();
        }
    }

    public SelectedTag getAggregationStrategy() {
        return new SelectedTag(AggregationStrategy, TAGS_AGGREGATION);
    }

    public void setAggregationStrategy(SelectedTag value) {
        if (value.getTags() == TAGS_AGGREGATION) {
            this.AggregationStrategy = value.getSelectedTag().getID();
        }
    }
   
    public String getAggregationStrategyName() {
        switch (AggregationStrategy) {
            case AGGREGATION_WEIGHTED_VOTE: return "Weighted Vote Aggregation";
            case AGGREGATION_MAJORITY_VOTE: return "Majority Vote Aggregation";
            case AGGREGATION_PLURALITY_VOTE: return "Plurality Vote Aggregation";
            case AGGREGATION_MIN_PLURALITY_THRESHOLD: return "Plurality Vote Aggregation With Minimum Threshold";
            case AGGREGATION_LOWEST_WEIGHT_VOTER: return "Lowest Weight Voter Classification";
            case AGGREGATION_TOP_K_VOTERS: return "Top-K Best Voters Classification";
            default: return "Unknown Nominal Classification Strategy";
        }
    }
    
    public int getTopKVoters() {
        return m_TopKVoters;
    }

    public void setTopKVoters(int numbTopKVotersVoters) {
        if (numbTopKVotersVoters < 1) {
            throw new IllegalArgumentException("Number of top voters must be at least 1");
        }
        m_TopKVoters = numbTopKVotersVoters;
    }

    public double getPluralityThreshold() {
        return m_PluralityThreshold;
    }

    public void setPluralityThreshold(double pluralityAgregationThreshold) {
        if (pluralityAgregationThreshold < 0.0 || pluralityAgregationThreshold > 100) {
            throw new IllegalArgumentException("Plurality threshold must be between 0 and 100");
        }
        m_PluralityThreshold = pluralityAgregationThreshold;
    }

    public double getWeightThreshold() {
        return m_WeightThreshold;
    }

    public void setWeightThreshold(double minWeightThreshold) {
        if (minWeightThreshold < 0 || minWeightThreshold > 100) {
            throw new IllegalArgumentException("Weight threshold must be between 0.0 and 1.0");
        }
        m_WeightThreshold = minWeightThreshold;
    }
    
    public void setConceptLearningStrategy(int strategy) {
        if (strategy >= CONCEPT_LEARNING_ALL_ALL && strategy <= CONCEPT_LEARNING_UNDER_THRESHOLD_MAJORITY) {
            this.ConceptLearningStrategy = strategy;
        } else {
            throw new IllegalArgumentException("Unknown Strategy : " + strategy);
        }
    }
    
    public String getConceptLearningStrategyName() {
        switch (ConceptLearningStrategy) {
            case CONCEPT_LEARNING_ALL_ALL: return "Closure of All Attributes & All Values";
            case CONCEPT_LEARNING_ALL_MAJORITY: return "Closure of All Attributes & Majority Value";
            case CONCEPT_LEARNING_PERTINENT_ALL: return "Closure of Pertinent Attributes & All Values";
            case CONCEPT_LEARNING_PERTINENT_MAJORITY: return "Closure of Pertinent Attributes & Majority Value";
            case CONCEPT_LEARNING_ABOVE_THRESHOLD_ALL: return "Closure of Attributes Above Threshold & All Values";
            case CONCEPT_LEARNING_ABOVE_THRESHOLD_MAJORITY: return "Closure of Attributes Above Threshold & Majority Value";
            case CONCEPT_LEARNING_UNDER_THRESHOLD_ALL: return "Closure of Attributes Under Threshold & All Values";
            case CONCEPT_LEARNING_UNDER_THRESHOLD_MAJORITY: return "Closure of Attributes Under Threshold & Majority Value";
            case N_ATTRIBUTES_COMBINATION_ALL_VALUES: return "Closure of N attributes combination All Values" ;
			case N_ATTRIBUTES_COMBINATION_MAJORITY_VALUES : return "Closure of N attributes combination Majority Values";
            default: return "Unknown Strategy";
        } 
    }
    
    public SelectedTag getConceptLearning() {
        return new SelectedTag(ConceptLearningStrategy, TAGS_ConceptLearning);
    }
    
    public void setConceptLearning(SelectedTag value) {
        if (value.getTags() == TAGS_ConceptLearning) {
            this.ConceptLearningStrategy = value.getSelectedTag().getID();
        }
    }

    public SelectedTag getPolicy_Selection() {
        return new SelectedTag(Policy_Selection, TAGS_Policy_Selection);
    }
    
    public void setPolicy_Selection(SelectedTag value) {
        if (value.getTags() == TAGS_Policy_Selection) {
            this.Policy_Selection = value.getSelectedTag().getID();
        }
    }

    public SelectedTag getThresholdTiming() {
        return new SelectedTag(m_ThresholdTiming, TAGS_THRESHOLD_TIMING);
    }

    public void setThresholdTiming(SelectedTag value) {
        if (value.getTags() == TAGS_THRESHOLD_TIMING) {
            m_ThresholdTiming = value.getSelectedTag().getID();
        }
    }
    
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
    public boolean getAllCombinations() {
        return allCombinations;
    }
    
    public void setAllCombinations(boolean value) {
        this.allCombinations = value;
        if (m_Debug) {
            System.out.println("[DEBUG] allCombinations set to: " + value);
        }
    }
    public int getCombinationSize() {
    	return combinationSize;
    }

    public void setCombinationSize(int n) {
    	combinationSize=n;
    }
    

    // Core methods
    @Override
    public TechnicalInformation getTechnicalInformation() {
        TechnicalInformation result = new TechnicalInformation(Type.INPROCEEDINGS);
        result.setValue(Field.YEAR, "2025");
        result.setValue(Field.AUTHOR, "Y. Souissi,N. MEDDOURI And F. Boissier");
        result.setValue(Field.PUBLISHER, "Academic Publisher");
        result.setValue(Field.ADDRESS, "Paris, France");
        result.setValue(Field.ORGANIZATION, "Research Unit on Programming, Algorithmics and Heuristics");
        result.setValue(Field.SCHOOL, "LRE EPITA");
        return result;
    }
    
    public String globalInfo() {
        return "Classifier Nominal Concepts (CNC) discovers classification Rules using concept learning "
             + "and feature measures.\n\n"
             + "For more information, see:\n\n"
             + getTechnicalInformation().toString() + "\n\nThis code was rewritten and improved by Yasmine Souissi, Nida Meddouri et Fabrice Boissier\n";
    }
    
    @Override
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
    
    @Override
    public Enumeration<Option> listOptions() {
        Vector<Option> newVector = new Vector<>(16);  

        newVector.addElement(new Option( 
            "\tCustom classifier key for display in Weka (not used internally).",
            "CNC_ID", 1, "-CNC_ID <string>"));

        // Options for nominal classification strategy
        newVector.addElement(new Option(
            "\t1: Weighted Vote Aggregation\n" +
            "\t2: Majority Vote Aggregation\n" +
            "\t3: Plurality Vote Aggregation\n" +
            "\t4: Minimum Plurality Threshold\n" +
            "\t5: Lowest Weight Voter\n" +
            "\t6: Top-K Best Voters",
            "aggregationStrategy", 1, "-aggregationStrategy <num>"));

        // Option for number of top voters
        newVector.addElement(new Option(
            "\tNumber of top voters to consider (for Top-K strategy)",
            "numbTopKVoters", 1, "-numbTopKVoters <num>"));

        // Option for plurality threshold
        newVector.addElement(new Option(
            "\tMinimum plurality threshold value (0.0-100)",
            "pluralityAgregationThreshold", 1, "-pluralityAgregationThreshold <num>"));

        // New option for weight threshold
        newVector.addElement(new Option(
            "\tWeight threshold for lowest weight voter (0.0-100)",
            "minWeightThreshold", 1, "-minWeightThreshold <num>"));

        // Option for concept learning strategy
        newVector.addElement(new Option(
            "\tConcept learning strategy:\n" +
            "\t1: All Attributes - All Values\n" +
            "\t2: All Attributes - Majority Value\n" +
            "\t3: Pertinent Attributes - All Values\n" +
            "\t4: Pertinent Attributes - Majority Value\n" +
            "\t5: Above Threshold - All Values\n" +
            "\t6: Above Threshold - Majority Value\n" +
            "\t7: Under Threshold - All Values\n" +
            "\t8: Under Threshold - Majority Value"+
            "\t9: N Attributes Combination- All Values\n"+
            "\t10: N Attributes Combination- Majority Values\n",

            "conceptStrategy", 1, "-conceptStrategy <num>"));

        // Option for threshold
        newVector.addElement(new Option(
            "\tThreshold value for attribute selection (>= 0).",
            "THRESHOLD", 1, "-THRESHOLD <num>"));
        
        newVector.addElement(new Option(
                "\tCombination size for attribute selection (>= 0).",
                "combinationSize", 1, "-combinationSize <num>"));
        
        newVector.addElement(new Option(
            "\tWhen to calculate threshold:\n" +
            "\t1: Before discretization\n" +
            "\t2: After discretization",
            "thresholdTiming", 1, "-thresholdTiming <1|2>"));
        
        // Option for auto threshold
        newVector.addElement(new Option(
            "\tIf True, threshold is auto-calculated (average of feature measures)",
            "autoThreshold", 1, "-autoThreshold <bool>"));
        
        // Option for ALL Combinations
        newVector.addElement(new Option(
            "\tIf True, Compute all attribute possible Combinations ",
            "allCombinations", 1, "-allCombinations <bool>"));
        
        // Option for feature measure
        newVector.addElement(new Option(
            "\tFeature selection measure:",
            "FeatureMeasure", 1, "-FeatureMeasure <entropy|infogain|gainratio>"));

        // Option for filter
        newVector.addElement(new Option(
            "\tFull filter specification:\n" +
            "\teg: \"weka.filters.unsupervised.attribute.Discretize -O -B 10 -M -1.0 -R first-last \"",
            "F", 1, "-F <filter specification>"));

        // Option for debug
        newVector.addElement(new Option(
            "\tEnable debug output",
            "debug", 0, "-debug"));

        // Parent class options
        newVector.addAll(Collections.list(super.listOptions()));

        return newVector.elements();
    }
    
    @Override
    public void setOptions(String[] options) throws Exception {
        // Reset default values
        ConceptLearningStrategy = CONCEPT_LEARNING_ALL_ALL;
        AggregationStrategy = AGGREGATION_WEIGHTED_VOTE;
        Utils.getOption("CNC_ID", options); // Just to consume the option

        // Handle aggregation strategy
        String aggregationStrategyStr = Utils.getOption("aggregationStrategy", options);
        if (!aggregationStrategyStr.isEmpty()) {
            int strategy = Integer.parseInt(aggregationStrategyStr);
            if (strategy < 1 || strategy > 6) {
                throw new IllegalArgumentException("Invalid aggregation strategy number. Must be between 1-6");
            }
            AggregationStrategy = strategy;
        }

        // Handle number of top voters
        String numbTopKVotersStr = Utils.getOption("numbTopKVoters", options);
        if (!numbTopKVotersStr.isEmpty()) {
            m_TopKVoters = Integer.parseInt(numbTopKVotersStr);
            if (m_TopKVoters < 1) {
                throw new IllegalArgumentException("Number of top voters must be at least 1");
            }
        }

     // Handle Combination size
        String combinationSizeStr = Utils.getOption("combinationSize", options);
        if (!combinationSizeStr.isEmpty()) {
            combinationSize = Integer.parseInt(combinationSizeStr);
            if (combinationSize < 1) {
                throw new IllegalArgumentException("Combination size must be at least 1");
            }
        }

        // Handle plurality threshold
        String pluralityAgregationThresholdStr = Utils.getOption("pluralityAgregationThreshold", options);
        if (!pluralityAgregationThresholdStr.isEmpty()) {
            m_PluralityThreshold = Double.parseDouble(pluralityAgregationThresholdStr);
            if (m_PluralityThreshold < 0 || m_PluralityThreshold > 100) {
                throw new IllegalArgumentException("Plurality threshold must be between 0 and 100");
            }
        }
        
        // Handle threshold timing
        String thresholdTimingStr = Utils.getOption("thresholdTiming", options);
        if (!thresholdTimingStr.isEmpty()) {
            int timing = Integer.parseInt(thresholdTimingStr);
            if (timing < 1 || timing > 2) {
                throw new IllegalArgumentException("Threshold timing must be 1 (before) or 2 (after)");
            }
            m_ThresholdTiming = timing;
        }

        // Handle weight threshold
        String minWeightThresholdStr = Utils.getOption("minWeightThreshold", options);
        if (!minWeightThresholdStr.isEmpty()) {
            m_WeightThreshold = Double.parseDouble(minWeightThresholdStr);
            if (m_WeightThreshold < 0 || m_WeightThreshold > 100) {
                throw new IllegalArgumentException("Weight threshold must be between 0 and 100");
            }
        }

        // Handle concept learning strategy
        String strategyStr = Utils.getOption("conceptStrategy", options);
        if (!strategyStr.isEmpty()) {
            int strategy = Integer.parseInt(strategyStr);
            if (strategy < 1 || strategy > 10) {
                throw new IllegalArgumentException("Invalid concept strategy number. Must be between 1-8");
            }
            ConceptLearningStrategy = strategy;
        }

        // Handle threshold
        String thresholdStr = Utils.getOption("THRESHOLD", options);
        if (!thresholdStr.isEmpty()) {
            double parsedThreshold = Double.parseDouble(thresholdStr);
            if (parsedThreshold < -1.0) {
                throw new IllegalArgumentException("Threshold must be -1 (auto) or >= 0.");
            }
            setThreshold(parsedThreshold);
        }
        
        // Handle auto threshold
        String autoThresholdStr = Utils.getOption("autoThreshold", options);
        if (!autoThresholdStr.isEmpty()) {
            setAuto_Threshold(autoThresholdStr.equalsIgnoreCase("true") || 
                            autoThresholdStr.equalsIgnoreCase("yes") ||
                            autoThresholdStr.equalsIgnoreCase("1"));
        }
        
        //allCombinations
        String allCombinationsStr = Utils.getOption("allCombinations", options);
        if (!allCombinationsStr.isEmpty()) {
           setAllCombinations(allCombinationsStr.equalsIgnoreCase("true") || 
        		   allCombinationsStr.equalsIgnoreCase("yes") ||
        		   allCombinationsStr.equalsIgnoreCase("1"));
        }

        // Handle feature measure
        String measureStr = Utils.getOption("FeatureMeasure", options);
        if (!measureStr.isEmpty()) {
            switch (measureStr.toLowerCase()) {
                case "entropy": FeatureMeasureMethod = FEAT_ENTROPY; break;
                case "infogain": FeatureMeasureMethod = FEAT_INFOGAIN; break;
                case "gainratio": FeatureMeasureMethod = FEAT_GAINRATIO; break;
                default: throw new IllegalArgumentException("Invalid feature measure: " + measureStr);
            }
        }

        // Handle filter
        String filterStr = Utils.getOption('F', options);
        if (!filterStr.isEmpty()) {
            String[] filterSpec = Utils.splitOptions(filterStr);
            if (filterSpec.length == 0) {
                setFilter(new weka.filters.unsupervised.attribute.Discretize());
            } else {
                try {
                    String filterName = filterSpec[0];
                    filterSpec[0] = "";
                    Filter filter = (Filter) Utils.forName(Filter.class, filterName, filterSpec);
                    setFilter(filter);
                } catch (Exception e) {
                    System.err.println("Warning: Could not set filter, using default Discretize. Error: " + e.getMessage());
                    setFilter(new weka.filters.unsupervised.attribute.Discretize());
                }
            }
        } else {
            setFilter(new weka.filters.unsupervised.attribute.Discretize());
        }

        // Handle debug
        m_Debug = Utils.getFlag("debug", options);

        super.setOptions(options);
    }

    @Override
    public String[] getOptions() {
        Vector<String> result = new Vector<>();
        
        // Custom key for Weka Experimenter
        result.add("-CNC_ID");
        result.add(getIdentifier());
        
        // Aggregation strategies
        result.add("-aggregationStrategy");
        result.add(String.valueOf(AggregationStrategy));

        // Number of top voters
        result.add("-numbTopKVoters");
        result.add(String.valueOf(m_TopKVoters));
        
        //combinationSize
        result.add("-combinationSize");
        result.add(String.valueOf(combinationSize));
        
        
        result.add("-thresholdTiming");
        result.add(String.valueOf(m_ThresholdTiming));

        // Plurality threshold
        result.add("-pluralityAgregationThreshold");
        result.add(String.valueOf(m_PluralityThreshold));

        // Weight threshold
        result.add("-minWeightThreshold");
        result.add(String.valueOf(m_WeightThreshold));

        // Concept learning strategy
        result.add("-conceptStrategy");
        result.add(String.valueOf(ConceptLearningStrategy));

        // Threshold
        result.add("-THRESHOLD");
        result.add(String.valueOf(getThreshold()));
        
        // Auto threshold
        result.add("-autoThreshold");
        result.add(String.valueOf(getAuto_Threshold()));
        
        //allCombinations
        result.add("-allCombinations");
        result.add(String.valueOf(getAllCombinations()));
        
        // Feature measure
        result.add("-FeatureMeasure");
        switch (FeatureMeasureMethod) {
            case FEAT_ENTROPY: result.add("entropy"); break;
            case FEAT_INFOGAIN: result.add("infogain"); break;
            case FEAT_GAINRATIO: result.add("gainratio"); break;
        }

        // Filter
        result.add("-F");
        result.add(getFilterSpec());

        // Debug
        if (m_Debug) {
            result.add("-debug");
        }

        // Parent class options
        Collections.addAll(result, super.getOptions());

        return result.toArray(new String[0]);
    }
    
    public String getIdentifier() {
        StringBuilder id = new StringBuilder();

        id.append("CNC");
        id.append("_Agg=").append(getAggregationStrategyName());

        if (AggregationStrategy == AGGREGATION_TOP_K_VOTERS) {
            id.append("_TopK=").append(m_TopKVoters);
        }

        if (AggregationStrategy == AGGREGATION_MIN_PLURALITY_THRESHOLD) {
            id.append("_PlurThr=").append(m_PluralityThreshold);
        }

        if (AggregationStrategy == AGGREGATION_LOWEST_WEIGHT_VOTER) {
            id.append("_MinWeigThr=").append(m_WeightThreshold);
        }

        id.append("_ConLearn=").append(getConceptLearningStrategyName());

        if (ConceptLearningStrategy >= CONCEPT_LEARNING_ABOVE_THRESHOLD_ALL &&
            ConceptLearningStrategy <= N_ATTRIBUTES_COMBINATION_MAJORITY_VALUES) {
            id.append("_Thr=").append(m_THRESHOLD);
            id.append("_ThrTim=").append(m_ThresholdTiming == 1 ? "BEF disc" : "AFT disc");
            id.append("_ThrCalc=").append(Auto_THRESHOLD ? "Auto" : "Manual");
        }

        id.append("_Measure=").append(featureMeasureToString(FeatureMeasureMethod));

        // Add filter information if used
        if (m_Filter != null) {
            id.append("_Filter=").append(getFilterTypeName());
            id.append("_FiltOpts=").append(getFilterOptionsAsString());
        }

        return id.toString();
    }

    private String getFilterTypeName() {
        if (m_Filter == null) {
            return "None";
        }
        return m_Filter.getClass().getSimpleName();
    }

    private String getFilterOptionsAsString() {
        if (m_Filter == null) {
            return "";
        }
        
        try {
            String[] options = m_Filter.getOptions();
            if (options == null || options.length == 0) {
                return "default";
            }
            
            // Build string with non-empty options
            StringBuilder opts = new StringBuilder();
            for (String opt : options) {
                if (!opt.trim().isEmpty()) {
                    if (opts.length() > 0) {
                        opts.append(" ");
                    }
                    opts.append(opt);
                }
            }
            return opts.toString();
        } catch (Exception e) {
            return "error";
        }
    }

    @Override
    public double classifyInstance(Instance inst) throws Exception {
        if (m_classifierNC == null) {
            //throw new IllegalStateException("Model not built. Call buildClassifier() first.");
        	return -1.0;
        }
        if (structure == null)
            throw new Exception("No input instance format defined");
        
        

        // Filter the instance
        m_Filter.input(inst);
        m_Filter.batchFinished();
        Instance filteredInst = m_Filter.output();
        Classify_Instance classifier = new Classify_Instance();
 
        double result = -1.0;
        int nb_class = Classification_Rule.getMaxNumClass(m_classifierNC);
        
        switch (AggregationStrategy) {
            case AGGREGATION_WEIGHTED_VOTE:
                result = classifier.classify_Instance_nom_VotePond(
                    filteredInst, m_classifierNC, nb_class, m_Debug);
                break;
                
            case AGGREGATION_MAJORITY_VOTE:
                result = classifier.classify_Instance_nom_VoteMaj(
                    filteredInst, m_classifierNC, nb_class, m_Debug);
                break;
                
            case AGGREGATION_PLURALITY_VOTE:
                result = classifier.classify_Instance_nom_VotePlur(
                    filteredInst, m_classifierNC, nb_class, m_Debug);
                break;
                
            case AGGREGATION_MIN_PLURALITY_THRESHOLD:
                result = classifier.classify_Instance_nom_VotePlurMin(
                    filteredInst, m_classifierNC, nb_class, m_PluralityThreshold, m_Debug);
                break;
                
            case AGGREGATION_LOWEST_WEIGHT_VOTER:
                result = classifier.classify_Instance_nom_VoteLowPondVoter(
                    filteredInst, m_classifierNC, nb_class, m_WeightThreshold, m_Debug);
                break;
                
            case AGGREGATION_TOP_K_VOTERS:
                result = classifier.classify_Instance_nom_VoteNumBestVoterS(
                    filteredInst, m_classifierNC, nb_class, m_TopKVoters, m_Debug);
                break;
                
            default:
                throw new Exception("Unknown nominal classification strategy");
        }
        
        return result == -1.0 ? Utils.missingValue() : result;
    }

    public double getm_Threshold(double[] featureMeasures) {
        if (Auto_THRESHOLD) {
            double sum = 0.0;
            for (double value : featureMeasures) {
                sum += value;
            }
            m_THRESHOLD = sum / featureMeasures.length;
        }
        return m_THRESHOLD;
    }

    
    
    // buildClassifier() builds the classifier using the training data, it is mandatory to implement this function

    
    public void buildClassifier(Instances data) throws Exception {

    	

        // Vérification des capacités et prétraitement des données
        getCapabilities().testWithFail(data);
        
        structure = new Instances(data, 0); // sauvegarde du format
        // Étape 1: Préparation des données
        m_FilteredInstances = new Instances(data);
        m_FilteredInstances.deleteWithMissingClass();
        
        int n=0;
        
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
       
        if (m_ThresholdTiming == THRESHOLD_BEFORE_DISCRETIZATION) {
        	m_THRESHOLD=getm_Threshold(extractor.computeFeatureMeasures(data));
        	
        }
        else if ( m_ThresholdTiming == THRESHOLD_AFTER_DISCRETIZATION) {
        	m_THRESHOLD=getm_Threshold(extractor.computeFeatureMeasures(m_FilteredInstances));

         }
    
        if (this.getAllCombinations()==true) {
        	switch (ConceptLearningStrategy) {
            case CONCEPT_LEARNING_ALL_ALL:
                if (m_Debug) {
                    System.out.println("=== ALL ATTRIBUTES - ALL VALUES - AllCombinations MODE ACTIVE ===");
                    System.out.println("Extraction using all attributes with all values");
                    System.out.println("Used Measure: " + featureMeasureToString(FeatureMeasureMethod));
                }
                m_classifierNC = extractor.extractAllAttributesAllValuesAllCombinations(m_FilteredInstances);
               
                break;
                
            case CONCEPT_LEARNING_ALL_MAJORITY:
                if (m_Debug) {
                    System.out.println("=== ALL ATTRIBUTES - MAJORITY VALUE - AllCombinations MODE ACTIVE ===");
                    System.out.println("Extraction using all attributes with majority values");
                    System.out.println("Used Measure: " + featureMeasureToString(FeatureMeasureMethod));
                }
                m_classifierNC = extractor.extractAllAttributesMajorityValueAllCombinations(m_FilteredInstances);
                if (m_Debug) {
                    printRuleStatistics();
                }
                break;
                
            case CONCEPT_LEARNING_PERTINENT_ALL:
                if (m_Debug) {
                    System.out.println("=== PERTINENT ATTRIBUTE - ALL VALUES - AllCombinations MODE ACTIVE ==="); 
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
                    System.out.println("=== PERTINENT ATTRIBUTE - MAJORITY VALUE - AllCombinations MODE ACTIVE ===");
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
                    System.out.println("=== ABOVE THRESHOLD - ALL VALUES - AllCombinations MODE ACTIVE ===");
                    if(Auto_THRESHOLD) {
                        System.out.println("===Auto Threshold Mode Active=== ");

                    }
                    if ( m_ThresholdTiming == THRESHOLD_BEFORE_DISCRETIZATION) {
                    	setThreshold(getm_Threshold(extractor.computeFeatureMeasures(data)));
                    	System.out.println("Threshold timing::THRESHOLD_BEFORE_DISCRETIZATION");

                    }
                    else if (m_ThresholdTiming == THRESHOLD_AFTER_DISCRETIZATION) {
                    	setThreshold(getm_Threshold(extractor.computeFeatureMeasures(m_FilteredInstances)));
                    	System.out.println("Threshold timing:: THRESHOLD_AFTER_DISCRETIZATION");

                    }
                    System.out.printf("Extraction with (Threshold=%.4f) " ,m_THRESHOLD);
                    System.out.println("Used Measure: " + featureMeasureToString(FeatureMeasureMethod));
                    extractor.printTopPercentageAttributes(m_FilteredInstances, m_THRESHOLD);

                }
                m_classifierNC = extractor.extractAttributeAboveThresholdAllValuesAllCombinations(m_FilteredInstances,m_THRESHOLD);
                if (m_Debug) {
                    printRuleStatistics();
                }

                break;
                 
            case CONCEPT_LEARNING_ABOVE_THRESHOLD_MAJORITY:
                if (m_Debug) {
                    System.out.println("=== ABOVE THRESHOLD - MAJORITY VALUES - AllCombinations MODE ACTIVE ===");
                    if(Auto_THRESHOLD) {
                        System.out.println("===Auto Threshold Mode Active=== ");

                    }
                    if ( m_ThresholdTiming == THRESHOLD_BEFORE_DISCRETIZATION) {
                    	System.out.println("Threshold timing::THRESHOLD_BEFORE_DISCRETIZATION");

                    }
                    else if (m_ThresholdTiming == THRESHOLD_AFTER_DISCRETIZATION) {
                    	System.out.println("Threshold timing:: THRESHOLD_AFTER_DISCRETIZATION");

                    }
                    
                    System.out.printf("Extraction with(Threshold=%.4f) " ,m_THRESHOLD);
                    System.out.println("Used Measure: " + featureMeasureToString(FeatureMeasureMethod));
                    extractor.printTopPercentageAttributes(m_FilteredInstances, m_THRESHOLD);

                }
                m_classifierNC = extractor.extractAttributeAboveThresholdMajorityValuesAllCombinations(m_FilteredInstances,m_THRESHOLD);
                if (m_Debug) {
                    printRuleStatistics();
                }

                break;
                
            case CONCEPT_LEARNING_UNDER_THRESHOLD_ALL:
                if (m_Debug) {
                    System.out.println("=== UNDER THRESHOLD - ALL VALUES - AllCombinations MODE ACTIVE ===");
                    if(Auto_THRESHOLD) {
                        System.out.println("===Auto Threshold Mode Active=== ");

                    }
                    if ( m_ThresholdTiming == THRESHOLD_BEFORE_DISCRETIZATION) {
                    	System.out.println("Threshold timing::THRESHOLD_BEFORE_DISCRETIZATION");

                    }
                    else if (m_ThresholdTiming == THRESHOLD_AFTER_DISCRETIZATION) {
                    	System.out.println("Threshold timing:: THRESHOLD_AFTER_DISCRETIZATION");

                    }
                    System.out.printf("Extraction with (Threshold=%.4f) " ,m_THRESHOLD);
                    System.out.println("Used Measure: " + featureMeasureToString(FeatureMeasureMethod));
                    extractor.printAttributesUnderThreshold(m_FilteredInstances, m_THRESHOLD);

                }
                m_classifierNC = extractor.extractAttributeUnderThresholdAllValuesAllCombinations
                		(m_FilteredInstances,m_THRESHOLD);
                if (m_Debug) {
                    printRuleStatistics();
                }

                break;
                
            case CONCEPT_LEARNING_UNDER_THRESHOLD_MAJORITY:
                if (m_Debug) {
                    System.out.println("=== UNDER THRESHOLD - MAJORITY VALUES - AllCombinations MODE ACTIVE ===");
                    if(Auto_THRESHOLD) {
                        System.out.println("===Auto Threshold Mode Active=== ");

                    }
                    if ( m_ThresholdTiming == THRESHOLD_BEFORE_DISCRETIZATION) {
                    	System.out.println("Threshold timing::THRESHOLD_BEFORE_DISCRETIZATION");

                    }
                    else if (m_ThresholdTiming == THRESHOLD_AFTER_DISCRETIZATION) {
                    	System.out.println("Threshold timing:: THRESHOLD_AFTER_DISCRETIZATION");

                    }
                    System.out.printf("Extraction with(Threshold=%.4f) " ,m_THRESHOLD);
                    System.out.println("Used Measure: " + featureMeasureToString(FeatureMeasureMethod));
                    extractor.printAttributesUnderThreshold(m_FilteredInstances, m_THRESHOLD);

                }
                m_classifierNC = extractor.extractAttributeUnderThresholdAllValuesAllCombinations(m_FilteredInstances,m_THRESHOLD);
                if (m_Debug) {
                    printRuleStatistics();
                }
                break;
            case N_ATTRIBUTES_COMBINATION_ALL_VALUES:
                if (m_Debug) {
                    System.out.println("=== N Attribute - ALL VALUES - AllCombinations MODE ACTIVE ===");
                }

                n = getCombinationSize();
                int nbAttr = m_FilteredInstances.numAttributes() - 1; // -1 pour la classe

                if (n > nbAttr) {
                    if (m_Debug) {
                        System.out.println(">> N > nombre d'attributs : aucune règle générée");
                    }
                    m_classifierNC = null; 
                    
                    break;
                }


                m_classifierNC = extractor.extractNAttributesAllValues(m_FilteredInstances, n);
                if (m_Debug) {
                    printRuleStatistics();
                }
                break;

            case N_ATTRIBUTES_COMBINATION_MAJORITY_VALUES:
                if (m_Debug) {
                    System.out.println("=== N Attribute - MAJORITY VALUES - AllCombinations MODE ACTIVE ===");
                }

                n = getCombinationSize();
                nbAttr = m_FilteredInstances.numAttributes() - 1;

                if (n > nbAttr) {
                    if (m_Debug) {
                        System.out.println(">> N > nombre d'attributs : aucune règle générée");
                    }
                    m_classifierNC = null; 

                    break;
                }

                m_classifierNC = extractor.extractNAttributesMajorityValues(m_FilteredInstances, n);
                if (m_Debug) {
                    printRuleStatistics();
                }
                break;

        }
        }
        else if(this.getAllCombinations()==false) {
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
                    System.out.println("=== PERTINENT ATTRIBUTE - MAJORITY VALUE MODE ACTIVE ===");
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
                    if(Auto_THRESHOLD) {
                        System.out.println("===Auto Threshold Mode Active=== ");

                    }
                    if ( m_ThresholdTiming == THRESHOLD_BEFORE_DISCRETIZATION) {
                    	setThreshold(getm_Threshold(extractor.computeFeatureMeasures(data)));
                    	System.out.println("Threshold timing::THRESHOLD_BEFORE_DISCRETIZATION");

                    }
                    else if (m_ThresholdTiming == THRESHOLD_AFTER_DISCRETIZATION) {
                    	setThreshold(getm_Threshold(extractor.computeFeatureMeasures(m_FilteredInstances)));
                    	System.out.println("Threshold timing:: THRESHOLD_AFTER_DISCRETIZATION");

                    }
                    System.out.printf("Extraction with (Threshold=%.4f) " ,m_THRESHOLD);
                    System.out.println("Used Measure: " + featureMeasureToString(FeatureMeasureMethod));
                    extractor.printTopPercentageAttributes(m_FilteredInstances, m_THRESHOLD);

                }
                m_classifierNC = extractor.extractAttributeAboveThresholdAllValues(m_FilteredInstances,m_THRESHOLD);
                if (m_Debug) {
                    printRuleStatistics();
                }

                break;
                
            case CONCEPT_LEARNING_ABOVE_THRESHOLD_MAJORITY:
                if (m_Debug) {
                    System.out.println("=== ABOVE THRESHOLD - MAJORITY VALUES MODE ACTIVE ===");
                    if(Auto_THRESHOLD) {
                        System.out.println("===Auto Threshold Mode Active=== ");

                    }
                    if ( m_ThresholdTiming == THRESHOLD_BEFORE_DISCRETIZATION) {
                    	System.out.println("Threshold timing::THRESHOLD_BEFORE_DISCRETIZATION");

                    }
                    else if (m_ThresholdTiming == THRESHOLD_AFTER_DISCRETIZATION) {
                    	System.out.println("Threshold timing:: THRESHOLD_AFTER_DISCRETIZATION");

                    }
                    
                    System.out.printf("Extraction with(Threshold=%.4f) " ,m_THRESHOLD);
                    System.out.println("Used Measure: " + featureMeasureToString(FeatureMeasureMethod));
                    extractor.printTopPercentageAttributes(m_FilteredInstances, m_THRESHOLD);

                }
                m_classifierNC = extractor.extractAttributeAboveThresholdMajorityValues(m_FilteredInstances,m_THRESHOLD);
                if (m_Debug) {
                    printRuleStatistics();
                }

                break;
                
            case CONCEPT_LEARNING_UNDER_THRESHOLD_ALL:
                if (m_Debug) {
                    System.out.println("=== UNDER THRESHOLD - ALL VALUES MODE ACTIVE ===");
                    if(Auto_THRESHOLD) {
                        System.out.println("===Auto Threshold Mode Active=== ");

                    }
                    if ( m_ThresholdTiming == THRESHOLD_BEFORE_DISCRETIZATION) {
                    	System.out.println("Threshold timing::THRESHOLD_BEFORE_DISCRETIZATION");

                    }
                    else if (m_ThresholdTiming == THRESHOLD_AFTER_DISCRETIZATION) {
                    	System.out.println("Threshold timing:: THRESHOLD_AFTER_DISCRETIZATION");

                    }
                    System.out.printf("Extraction with (Threshold=%.4f) " ,m_THRESHOLD);
                    System.out.println("Used Measure: " + featureMeasureToString(FeatureMeasureMethod));
                    extractor.printAttributesUnderThreshold(m_FilteredInstances, m_THRESHOLD);

                }
                m_classifierNC = extractor.extractAttributeUnderThresholdAllValues
                		(m_FilteredInstances,m_THRESHOLD);
                if (m_Debug) {
                    printRuleStatistics();
                }

                break;
                
            case CONCEPT_LEARNING_UNDER_THRESHOLD_MAJORITY:
                if (m_Debug) {
                    System.out.println("=== UNDER THRESHOLD - MAJORITY VALUES MODE ACTIVE ===");
                    if(Auto_THRESHOLD) {
                        System.out.println("===Auto Threshold Mode Active=== ");

                    }
                    if ( m_ThresholdTiming == THRESHOLD_BEFORE_DISCRETIZATION) {
                    	System.out.println("Threshold timing::THRESHOLD_BEFORE_DISCRETIZATION");

                    }
                    else if (m_ThresholdTiming == THRESHOLD_AFTER_DISCRETIZATION) {
                    	System.out.println("Threshold timing:: THRESHOLD_AFTER_DISCRETIZATION");

                    }
                    System.out.printf("Extraction with(Threshold=%.4f) " ,m_THRESHOLD);
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

        List<FormalConcept> concepts=extractor.m_GaloisConnection.getConceptList();
        
        this.conceptsListStr=extractor.m_GaloisConnection.getConceptLevelsAsString(concepts);

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
        sb.append(String.format("%-25s: %s\n", "Threshold ",autoThresholdToString()));
        sb.append(String.format("%-25s: %s\n", "Threshold Calculation",m_ThresholdTimingToString()));
        sb.append(String.format("%-25s: %.4f\n", "Threshold", m_THRESHOLD));  // Affichage du seuil avec 4 décimales
        sb.append(String.format("%-25s: %s\n", "Policy Selection", getPolicySelectionName()));
        sb.append(String.format("%-25s: %s\n", "Aggregation Strategy", getAggregationStrategyName()));
        sb.append(String.format("%-25s: %.4f\n", "Pularity Threshold", m_PluralityThreshold));  // Affichage du seuil avec 4 décimales
        sb.append(String.format("%-25s: %.4f\n", "Weight Threshold",m_WeightThreshold));  // Affichage du seuil avec 4 décimales
        sb.append(String.format("%-25s: %d\n", "TOP K Voters Threshold",m_TopKVoters));  

        
        
 
        sb.append("\n");
        
        sb.append("=== Concepts ===\n");
        sb.append(conceptsListStr).append("\n\n");
        

        
        // Statistiques de performance
        if (m_classifierNC != null && !m_classifierNC.isEmpty()) {
            sb.append("=== Performance ===\n");
           
            sb.append(String.format("%-25s: %d\n", "Number of Rules Generated", m_classifierNC.size()));
            sb.append("\n");
            
            // Détail des règles
            sb.append("=== Generated Rules ===\n");
            for (int i = 0; i < m_classifierNC.size(); i++) {
                sb.append(String.format("\nRule %d:  ", i + 1));
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