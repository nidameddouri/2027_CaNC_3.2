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
    public static final int aboveUnderPertThreshold_BEFORE_DISCRETIZATION = 1;
    public static final int aboveUnderPertThreshold_AFTER_DISCRETIZATION = 2;
    
    // Aggregation Voters constants
    public static final int AGGREGATION_WEIGHTED_VOTE = 1;  
    public static final int AGGREGATION_MAJORITY_VOTE = 2;  
    public static final int AGGREGATION_PLURALITY_VOTE = 3;  
    public static final int AGGREGATION_MIN_PLURALITY_aboveUnderPertThreshold = 4;  
    public static final int AGGREGATION_LOWEST_WEIGHT_VOTER = 5;  
    public static final int AGGREGATION_TOP_K_VOTERS = 6;  

    // Concept learning strategy constants
    public static final int CONCEPT_LEARNING_ALL_ALL = 1;
    public static final int CONCEPT_LEARNING_ALL_MAJORITY = 2;
    public static final int CONCEPT_LEARNING_PERTINENT_ALL = 3;
    public static final int CONCEPT_LEARNING_PERTINENT_MAJORITY = 4;
    public static final int CONCEPT_LEARNING_ABOVE_aboveUnderPertThreshold_ALL = 5;
    public static final int CONCEPT_LEARNING_ABOVE_aboveUnderPertThreshold_MAJORITY = 6;
    public static final int CONCEPT_LEARNING_UNDER_aboveUnderPertThreshold_ALL = 7;
    public static final int CONCEPT_LEARNING_UNDER_aboveUnderPertThreshold_MAJORITY = 8;
    public static final int N_ATTRIBUTES_COMBINATION_ALL_VALUES= 9;
    public static final int N_ATTRIBUTES_COMBINATION_MAJORITY_VALUES= 10;
    public static final int PERCENT_ATTRIBUTES_COMBINATION_ALL_VALUES= 11;
    public static final int PERCENT_ATTRIBUTES_COMBINATION_ALL_VALUES_WITH_VAR= 12;


    

    // Policy selection constants
    public static final int POLICY_FIRST = 1;
    public static final int POLICY_LAST = 2;
    
    private Instances structure;
    
    //All Combinations constants

    public static final boolean ALL_COMBINATIONS = true;

    //Concepts list
    String conceptsListStr;
    
    // Tag arrays for GUI options
    public static final Tag[] TAGS_aboveUnderPertThreshold_TIMING = {
        new Tag(aboveUnderPertThreshold_BEFORE_DISCRETIZATION, "Calculate threshold BEFORE discretization"),
        new Tag(aboveUnderPertThreshold_AFTER_DISCRETIZATION, "Calculate threshold AFTER discretization")
    };
    
    public static final Tag[] TAGS_PertinenceMeasure = {
        new Tag(FEAT_ENTROPY, "Entropy"),
        new Tag(FEAT_INFOGAIN, "Information Gain"),
        new Tag(FEAT_GAINRATIO, "Gain Ratio"),
    };
    
    public static final Tag[] TAGS_AGGREGATION = {
        new Tag(AGGREGATION_WEIGHTED_VOTE, "Weighted Vote "),
        new Tag(AGGREGATION_MAJORITY_VOTE, "Majority Vote "),
        new Tag(AGGREGATION_PLURALITY_VOTE, "Plurality Vote "),
        new Tag(AGGREGATION_MIN_PLURALITY_aboveUnderPertThreshold, "Plurality Vote  With Minimum Threshold"),
        new Tag(AGGREGATION_LOWEST_WEIGHT_VOTER, "Weighted Vote  With Minimum Weight Threshold"),
        new Tag(AGGREGATION_TOP_K_VOTERS, "Weighted Vote  With with Top-K Best Voters")
    };
    
    public static final Tag[] TAGS_CLOSURE_STRATEGY = {
        new Tag(CONCEPT_LEARNING_ALL_ALL, "Closure of All Attributes & All Values"),
        new Tag(CONCEPT_LEARNING_ALL_MAJORITY, "Closure of ALL attributes & Majority Value"),
        new Tag(CONCEPT_LEARNING_PERTINENT_ALL, "Closure of the Pertinent Attribute & All Values"),
        new Tag(CONCEPT_LEARNING_PERTINENT_MAJORITY, "Closure of the Pertinent attribute & Majority Value"),
        new Tag(CONCEPT_LEARNING_ABOVE_aboveUnderPertThreshold_ALL, "Closure of attributes above aboveUnderPertThreshold & All Values"),
        new Tag(CONCEPT_LEARNING_ABOVE_aboveUnderPertThreshold_MAJORITY, "Closure of attributes above aboveUnderPertThreshold & Majority Value"),
        new Tag(CONCEPT_LEARNING_UNDER_aboveUnderPertThreshold_ALL, "Closure of attributes under aboveUnderPertThreshold & All Values"),
        new Tag(CONCEPT_LEARNING_UNDER_aboveUnderPertThreshold_MAJORITY, "Closure of attributes under aboveUnderPertThreshold & Majority Value"),
        new Tag(N_ATTRIBUTES_COMBINATION_ALL_VALUES,"Closure of N attributes combination All Values" ),
        new Tag(N_ATTRIBUTES_COMBINATION_MAJORITY_VALUES,"Closure of N attributes combination Majority  Values" ),
        new Tag(PERCENT_ATTRIBUTES_COMBINATION_ALL_VALUES,"Closure of Percentage attributes combination All  Values" ),
        new Tag(PERCENT_ATTRIBUTES_COMBINATION_ALL_VALUES_WITH_VAR,"Closure of Percentage attributes combination with variance All  Values" ),


    };
    
    public static final Tag[] TAGS_policySelectionOccurence = {
        new Tag(POLICY_FIRST, "First occurrence in case of equality"),
        new Tag(POLICY_LAST, "Last occurrence in case of equality"),
    };

    // Class members
    protected int m_CNC = 0;
    protected double m_LowPondVoter = 66.66;
    protected double m_aboveUnderPertThreshold = 0;
    protected boolean autoPertThresholdComb = true;
    protected int m_beforeAfterDisc = aboveUnderPertThreshold_BEFORE_DISCRETIZATION;
    protected int PertinenceMeasureMethod = FEAT_GAINRATIO;
    protected int aggregationVoters = AGGREGATION_WEIGHTED_VOTE;
    protected int closureStrategy = CONCEPT_LEARNING_ALL_ALL;
    protected int policySelectionOccurence = POLICY_LAST;
    protected int m_TopKVoters = 10;
    protected double m_PluralityVotersThreshold = 50;
    protected double m_WeightedVoteThreshold = 10;
    protected boolean fullAttribCombinations = true;
    protected int attributesCombinSize = 1;
    protected float attributesCombinPercent = 1;
    protected float varianceAttriCombPercent = 1;
    protected String vectorialConceptList="";
    protected boolean m_Debug = false;



    
    protected ReplaceMissingValues m_Missing = new ReplaceMissingValues();
    protected Classifier m_ZeroR;
    protected static Filter m_Filter = new weka.filters.unsupervised.attribute.Discretize();
    protected Instances m_FilteredInstances;
    public static ArrayList<Classification_Rule> m_classifierNC;
    
    protected static InfoGainAttributeEval m_InfoGainAttributeEval = new InfoGainAttributeEval();
    protected static GainRatioAttributeEval m_GainRatioAttributeEval = new GainRatioAttributeEval();
    
    Calendar calendar;
    SimpleDateFormat sdf = new SimpleDateFormat("E MM/dd/yyyy HH:mm:ss.SSS");
    
    public float getAttributesCombinPercent() {
        return attributesCombinPercent;
    }

    public void setAttributesCombinPercent(float attributesCombinPercent) {
        if (attributesCombinPercent < 0 || attributesCombinPercent > 100) {
            throw new IllegalArgumentException("attributesCombinPercent must be between 0 and 100");
        }
        this.attributesCombinPercent = attributesCombinPercent;
    }


    public float getVarianceAttriCombPercent() {
        return varianceAttriCombPercent;
    }

    public void setVarianceAttriCombPercent(float value) {
        this.varianceAttriCombPercent = value;
    }

  //TIPTEXT functions

    public String CNCTipText() {
        return "Number of CNC classifiers to use.";
    }

    public String aboveUnderPertThresholdTipText() {
        return "Minimum relevance (in %) threshold for attribute selection-must be between 0 and 1 (used with Closure of atrributes Above/Under threshold).";
    }

    public String autoPertThresholdCombTipText() {
        return "If enabled, automatically determines the optimal threshold (used with Closure of atrributes Above/Under threshold).";
    }

    public String beforeAfterDiscTipText() {
        return "When to apply the threshold: before or after discretization (used with Closure of atrributes Above/Under threshold).";
    }

    public String pertinenceMeasureTipText() {
        return "Measure used to evaluate the relevance of attributes (e.g., Gain Ratio, Info Gain).";
    }

    public String aggregationVotersTipText() {
        return "Strategy used to aggregate the votes from selected concepts.";
    }

    public String closureStrategyTipText() {
    	return "Closure strategy for concept leaning";
    }

    public String policySelectionOccurenceOccurenceTipText() {
        return "Policy used to select the final concept in case of conflicts.";
    }

    public String topKVotersTipText() {
        return "Number of top voters to keep in the aggregation (used with Weighted Vote Aggregation With Top-K Best Voters).";
    }

    public String pluralityVotersThresholdTipText() {
        return "Minimum percentage of agreement among voters to accept a decision (used with Plurality Vote Aggregation With Minimum Threshold).";
    }

    public String weightedVoteThresholdTipText() {
        return "Minimum weight required for a voter to be considered (used with Weighted Vote Aggregation With Minimum Weight Threshold).";
    }

    public String fullAttribCombinationTipText() {
        return "If enabled, tests all possible combinations of attributes (used with all learning strategies).";
    }

    public String attributesCombinSizeTipText() {
        return "Size of attribute combinations (used with Closure of N attributes combination).";
    }

    public String attributesCombinPercentTipText() {
        return "Percentage of attributes to select for each combination (used with Closure of Percentage attributes combination).";
    }

    public String varianceAttriCombPercentTipText() {
        return "Variance to apply around the selected attribute percentage (used with Closure of Percentage attributes combination with variance).";
    }

    public String vectorialConceptListTipText() {
        return "list of concepts in vector format to use directly.";
    }

    public String filterTipText() {
        return "Filter to apply to the dataset before classification.";
    }




    // Getters and Setters
    public double getAboveUnderPertThreshold() {
        return m_aboveUnderPertThreshold;
    }

    public void setAboveUnderPertThreshold(double t) {
        if (t < 0.0 || t > 1.0) {
            throw new IllegalArgumentException(
                "Threshold must be between 0.0 and 1.0.");
        }
        this.m_aboveUnderPertThreshold = t;
    }


    
    public boolean getautoPertThresholdComb() {
        return autoPertThresholdComb;
    }

    public void setautoPertThresholdComb(boolean s) {
        autoPertThresholdComb = s;  
    }
    
    public String autoPertThresholdCombToString() {
        return autoPertThresholdComb ? "Auto Generated" : "Manually Set";
    }
    
    public String m_beforeAfterDiscToString() {
        return (m_beforeAfterDisc == 1) ? "BEFORE discretization" : "AFTER discretization";
    }

    public SelectedTag getPertinenceMeasure() {
        return new SelectedTag(PertinenceMeasureMethod, TAGS_PertinenceMeasure);
    }
    
    public void setPertinenceMeasure(SelectedTag value) {
        if (value.getTags() == TAGS_PertinenceMeasure) {
            this.PertinenceMeasureMethod = value.getSelectedTag().getID();
        }
    }

    public SelectedTag getaggregationVoters() {
        return new SelectedTag(aggregationVoters, TAGS_AGGREGATION);
    }

    public void setaggregationVoters(SelectedTag value) {
        if (value.getTags() == TAGS_AGGREGATION) {
            this.aggregationVoters = value.getSelectedTag().getID();
        }
    }
   
    public String getaggregationVotersName() {
        switch (aggregationVoters) {
            case AGGREGATION_WEIGHTED_VOTE: return "Weighted Vote Aggregation";
            case AGGREGATION_MAJORITY_VOTE: return "Majority Vote Aggregation";
            case AGGREGATION_PLURALITY_VOTE: return "Plurality Vote Aggregation";
            case AGGREGATION_MIN_PLURALITY_aboveUnderPertThreshold: return "Plurality Vote Aggregation With Minimum Threshold";
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

    public double getPluralityVotersThreshold() {
        return m_PluralityVotersThreshold;
    }

    public void setPluralityVotersThreshold(double pluralityAgregationThreshold) {
        if (pluralityAgregationThreshold < 0.0 || pluralityAgregationThreshold > 100) {
            throw new IllegalArgumentException("Plurality threshold must be between 0 and 100");
        }
        m_PluralityVotersThreshold = pluralityAgregationThreshold;
    }

    public double getWeightedVoteThreshold() {
        return m_WeightedVoteThreshold;
    }

    public void setWeightedVoteThreshold(double minWeightedVoteThreshold) {
        if (minWeightedVoteThreshold < 0 || minWeightedVoteThreshold > 100) {
            throw new IllegalArgumentException("Weight threshold must be between 0.0 and 1.0");
        }
        m_WeightedVoteThreshold = minWeightedVoteThreshold;
    }
    
    public void setClosureStrategy(SelectedTag value) {
        if (value.getTags() == TAGS_CLOSURE_STRATEGY) {
            this.closureStrategy = value.getSelectedTag().getID();
        }
    }

    
    public String getClosureStrategyName() {
        switch (closureStrategy) {
            case CONCEPT_LEARNING_ALL_ALL: return "Closure of All Attributes & All Values";
            case CONCEPT_LEARNING_ALL_MAJORITY: return "Closure of All Attributes & Majority Value";
            case CONCEPT_LEARNING_PERTINENT_ALL: return "Closure of Pertinent Attributes & All Values";
            case CONCEPT_LEARNING_PERTINENT_MAJORITY: return "Closure of Pertinent Attributes & Majority Value";
            case CONCEPT_LEARNING_ABOVE_aboveUnderPertThreshold_ALL: return "Closure of Attributes Above Threshold & All Values";
            case CONCEPT_LEARNING_ABOVE_aboveUnderPertThreshold_MAJORITY: return "Closure of Attributes Above Threshold & Majority Value";
            case CONCEPT_LEARNING_UNDER_aboveUnderPertThreshold_ALL: return "Closure of Attributes Under Threshold & All Values";
            case CONCEPT_LEARNING_UNDER_aboveUnderPertThreshold_MAJORITY: return "Closure of Attributes Under Threshold & Majority Value";
            case N_ATTRIBUTES_COMBINATION_ALL_VALUES: return "Closure of N attributes combination All Values" ;
			case N_ATTRIBUTES_COMBINATION_MAJORITY_VALUES : return "Closure of N attributes combination Majority Values";
			case PERCENT_ATTRIBUTES_COMBINATION_ALL_VALUES : return "Closure of percentage attributes combination All Values";
			case PERCENT_ATTRIBUTES_COMBINATION_ALL_VALUES_WITH_VAR : return "Closure of percentage attributes combination with variance All Values";

			
            default: return "Unknown Strategy";
        } 
    }
    
    public SelectedTag getClosureStrategy() {
        return new SelectedTag(closureStrategy, TAGS_CLOSURE_STRATEGY);
        
    }
    
    public void getClosureStrategy(SelectedTag value) {
        if (value.getTags() == TAGS_CLOSURE_STRATEGY) {
            this.closureStrategy = value.getSelectedTag().getID();
        }
    }

    public SelectedTag getpolicySelectionOccurence() {
        return new SelectedTag(policySelectionOccurence, TAGS_policySelectionOccurence);
    }
    
    public void setpolicySelectionOccurence(SelectedTag value) {
        if (value.getTags() == TAGS_policySelectionOccurence) {
            this.policySelectionOccurence = value.getSelectedTag().getID();
        }
    }

    public SelectedTag getbeforeAfterDisc() {
        return new SelectedTag(m_beforeAfterDisc, TAGS_aboveUnderPertThreshold_TIMING);
    }

    public void setbeforeAfterDisc(SelectedTag value) {
        if (value.getTags() == TAGS_aboveUnderPertThreshold_TIMING) {
            m_beforeAfterDisc = value.getSelectedTag().getID();
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
    public boolean getfullAttribCombinations() {
        return fullAttribCombinations;
    }
    
    public void setfullAttribCombinations(boolean value) {
        this.fullAttribCombinations = value;
       
    }
    public int getAttributesCombinSize( ) {
    	return attributesCombinSize;
    }

    public void setAttributesCombinSize(int n) {
    	attributesCombinSize=n;
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
        Vector<Option> newVector = new Vector<>(20);

        // Custom classifier key
        newVector.addElement(new Option(
            "\tCustom classifier key for display in Weka (not used internally).",
            "CNC_ID", 1, "-CNC_ID <string>"));

        // Aggregation voters
        newVector.addElement(new Option(
            "\t1: Weighted Vote Aggregation\n" +
            "\t2: Majority Vote Aggregation\n" +
            "\t3: Plurality Vote Aggregation\n" +
            "\t4: Minimum Plurality Threshold\n" +
            "\t5: Lowest Weight Voter\n" +
            "\t6: Top-K Best Voters",
            "aggregationVoters", 1, "-aggregationVoters <num>"));

        // Number of top voters
        newVector.addElement(new Option(
            "\tNumber of top voters to consider (for Top-K strategy)",
            "numbTopKVoters", 1, "-numbTopKVoters <num>"));

        // Plurality aggregation threshold
        newVector.addElement(new Option(
            "\tMinimum plurality threshold value (0.0-100)",
            "pluralityAgregationThreshold", 1, "-pluralityAgregationThreshold <num>"));

        // Minimum weighted vote threshold
        newVector.addElement(new Option(
            "\tWeight threshold for lowest weight voter (0.0-100)",
            "minWeightedVoteThreshold", 1, "-minWeightedVoteThreshold <num>"));

        // Closure strategy
        newVector.addElement(new Option(
            "\tClosure strategy:\n" +
            "\t1: All Attributes - All Values\n" +
            "\t2: All Attributes - Majority Value\n" +
            "\t3: Pertinent Attributes - All Values\n" +
            "\t4: Pertinent Attributes - Majority Value\n" +
            "\t5: Above Threshold - All Values\n" +
            "\t6: Above Threshold - Majority Value\n" +
            "\t7: Under Threshold - All Values\n" +
            "\t8: Under Threshold - Majority Value\n" +
            "\t9: N Attributes Combination - All Values\n" +
            "\t10: N Attributes Combination - Majority Values\n" +
            "\t11: Percent Attributes Combination - All Values\n" +
            "\t12: Percent Attributes Combination With Variance - All Values\n",
            "closureStrategy", 1, "-closureStrategy <num>"));

        // Threshold for above/under pertinent attributes
        newVector.addElement(new Option(
            "\tThreshold value for attribute selection (0.0 - 1.0).",
            "aboveUnderPertThreshold", 1, "-aboveUnderPertThreshold <num>"));

        // Attribute combination size
        newVector.addElement(new Option(
            "\tCombination size for attribute selection (>= 0).",
            "attributesCombinSize", 1, "-attributesCombinSize <num>"));

        // Attribute combination percent
        newVector.addElement(new Option(
            "\tPercentage of attributes to use in combination (0.0 - 100.0)",
            "attributesCombinPercent", 1, "-attributesCombinPercent <float>"));

        // Variance for combination percent
        newVector.addElement(new Option(
            "\tVariance for the attribute percentage (used when closureStrategy=12)",
            "varianceAttriCombPercent", 1, "-varianceAttriCombPercent <float>"));

        // Before/after discretization
        newVector.addElement(new Option(
            "\tWhen to calculate threshold:\n" +
            "\t1: Before discretization\n" +
            "\t2: After discretization",
            "beforeAfterDisc", 1, "-beforeAfterDisc <1|2>"));

        // Auto pertinent threshold combination
        newVector.addElement(new Option(
            "\tIf True, threshold is auto-calculated (average of feature measures)",
            "autoPertThresholdComb", 1, "-autoPertThresholdComb <bool>"));

        // Full attribute combinations
        newVector.addElement(new Option(
            "\tIf True, compute all possible attribute combinations",
            "fullAttribCombinations", 1, "-fullAttribCombinations <bool>"));

        // Pertinence measure
        newVector.addElement(new Option(
            "\tFeature selection measure: entropy | infogain | gainratio",
            "PertinenceMeasure", 1, "-PertinenceMeasure <string>"));

        // Policy selection occurrence
        newVector.addElement(new Option(
            "\tPolicy selection occurrence in case of equality (int value)",
            "policySelectionOccurence", 1, "-policySelectionOccurence <num>"));

        // Filter specification
        newVector.addElement(new Option(
            "\tFull filter specification:\n" +
            "\teg: \"weka.filters.unsupervised.attribute.Discretize -O -B 10 -M -1.0 -R first-last\"",
            "F", 1, "-F <filter specification>"));

        // Option for debug
        newVector.addElement(new Option(
            "\tEnable debug output",
            "debug", 0, "-debug"));


        // Add parent options
        newVector.addAll(Collections.list(super.listOptions()));

        return newVector.elements();
    }

    @Override
    public void setOptions(String[] options) throws Exception {
        // Default values
        closureStrategy = CONCEPT_LEARNING_ALL_ALL;
        aggregationVoters = AGGREGATION_WEIGHTED_VOTE;

        // Consume CNC_ID
        Utils.getOption("CNC_ID", options);

        // Aggregation voters
        String s = Utils.getOption("aggregationVoters", options);
        if (!s.isEmpty()) aggregationVoters = Integer.parseInt(s);

        // Top-K voters
        s = Utils.getOption("numbTopKVoters", options);
        if (!s.isEmpty()) m_TopKVoters = Integer.parseInt(s);

        // Plurality threshold
        s = Utils.getOption("pluralityAgregationThreshold", options);
        if (!s.isEmpty()) m_PluralityVotersThreshold = Double.parseDouble(s);

        // Weighted vote threshold
        s = Utils.getOption("minWeightedVoteThreshold", options);
        if (!s.isEmpty()) m_WeightedVoteThreshold = Double.parseDouble(s);

        // Closure strategy
        s = Utils.getOption("closureStrategy", options);
        if (!s.isEmpty()) closureStrategy = Integer.parseInt(s);

        // Above/Under Pertinent Threshold
        s = Utils.getOption("aboveUnderPertThreshold", options);
        if (!s.isEmpty()) {
            double parsedThreshold = Double.parseDouble(s);
            if (parsedThreshold < 0.0 || parsedThreshold > 1.0) {
                throw new IllegalArgumentException(
                    "Threshold must be between 0.0 and 1.0.");
            }
            setAboveUnderPertThreshold(parsedThreshold);
        }


        // Combination size
        s = Utils.getOption("attributesCombinSize", options);
        if (!s.isEmpty()) attributesCombinSize = Integer.parseInt(s);

        // Combination percent
        s = Utils.getOption("attributesCombinPercent", options);
        if (!s.isEmpty()) attributesCombinPercent = Float.parseFloat(s);

        // Variance percent
        s = Utils.getOption("varianceAttriCombPercent", options);
        if (!s.isEmpty()) setVarianceAttriCombPercent(Float.parseFloat(s));

        // Before/after discretization
        s = Utils.getOption("beforeAfterDisc", options);
        if (!s.isEmpty()) m_beforeAfterDisc = Integer.parseInt(s);

        // Auto Pertinent Threshold
        s = Utils.getOption("autoPertThresholdComb", options);
        if (!s.isEmpty()) setautoPertThresholdComb(s.equalsIgnoreCase("true") || s.equalsIgnoreCase("yes") || s.equals("1"));

        // Full attribute combinations
        s = Utils.getOption("fullAttribCombinations", options);
        if (!s.isEmpty()) setfullAttribCombinations(s.equalsIgnoreCase("true") || s.equalsIgnoreCase("yes") || s.equals("1"));

        // Pertinence measure
        s = Utils.getOption("PertinenceMeasure", options);
        if (!s.isEmpty()) {
            switch (s.toLowerCase()) {
                case "entropy": PertinenceMeasureMethod = FEAT_ENTROPY; break;
                case "infogain": PertinenceMeasureMethod = FEAT_INFOGAIN; break;
                case "gainratio": PertinenceMeasureMethod = FEAT_GAINRATIO; break;
                default: throw new IllegalArgumentException("Invalid feature measure: " + s);
            }
        }

        // Policy selection occurrence
        s = Utils.getOption("policySelectionOccurence", options);
        if (!s.isEmpty()) policySelectionOccurence = Integer.parseInt(s);

       // Filter
        String filterStr = Utils.getOption("F", options); // <-- CORRECT : String "F"
        if (!filterStr.isEmpty()) {
            String[] filterSpec = Utils.splitOptions(filterStr);
            if (filterSpec.length == 0) {
                setFilter(new weka.filters.unsupervised.attribute.Discretize());
            } else {
                Filter f = (Filter) Utils.forName(Filter.class, filterSpec[0],
                                  Arrays.copyOfRange(filterSpec, 1, filterSpec.length));
                setFilter(f);
            }
        } else {
            setFilter(new weka.filters.unsupervised.attribute.Discretize());
        }


        
        // Debug flag
        m_Debug = Utils.getFlag("debug", options);

        // Parent options
        super.setOptions(options);
    }

    @Override
    public String[] getOptions() {
        Vector<String> result = new Vector<>();

        // Aggregation voters
        result.add("-aggregationVoters");
        result.add(String.valueOf(aggregationVoters));

        // Top-K voters
        result.add("-numbTopKVoters");
        result.add(String.valueOf(m_TopKVoters));

        // Plurality threshold
        result.add("-pluralityAgregationThreshold");
        result.add(String.valueOf(m_PluralityVotersThreshold));

        // Weighted vote threshold
        result.add("-minWeightedVoteThreshold");
        result.add(String.valueOf(m_WeightedVoteThreshold));

        // Closure strategy
        result.add("-closureStrategy");
        result.add(String.valueOf(closureStrategy));

        // Above/Under Pertinent Threshold
        result.add("-aboveUnderPertThreshold");
        result.add(String.valueOf(getAboveUnderPertThreshold()));

        // Combination size
        result.add("-attributesCombinSize");
        result.add(String.valueOf(attributesCombinSize));

        // Combination percent
        result.add("-attributesCombinPercent");
        result.add(String.valueOf(attributesCombinPercent));

        // Variance percent
        result.add("-varianceAttriCombPercent");
        result.add(String.valueOf(getVarianceAttriCombPercent()));

        // Before/after discretization
        result.add("-beforeAfterDisc");
        result.add(String.valueOf(m_beforeAfterDisc));

        // Auto Pertinent Threshold
        result.add("-autoPertThresholdComb");
        result.add(String.valueOf(getautoPertThresholdComb()));

        // Full attribute combinations
        result.add("-fullAttribCombinations");
        result.add(String.valueOf(getfullAttribCombinations()));

        // Pertinence measure
        result.add("-PertinenceMeasure");
        switch (PertinenceMeasureMethod) {
            case FEAT_ENTROPY: result.add("entropy"); break;
            case FEAT_INFOGAIN: result.add("infogain"); break;
            case FEAT_GAINRATIO: result.add("gainratio"); break;
        }

        // Policy selection occurrence
        result.add("-policySelectionOccurence");
        result.add(String.valueOf(policySelectionOccurence));

        // Filter
        result.add("-F");
        result.add(getFilterSpec());

        // Debug flag
        if (m_Debug) result.add("-debug");

        // Parent options
        Collections.addAll(result, super.getOptions());

        return result.toArray(new String[0]);
    }

    
    public String getIdentifier() {
        StringBuilder id = new StringBuilder();

        id.append("CNC");
        id.append("_Agg=").append(getaggregationVotersName());

        if (aggregationVoters == AGGREGATION_TOP_K_VOTERS) {
            id.append("_TopK=").append(m_TopKVoters);
        }

        if (aggregationVoters == AGGREGATION_MIN_PLURALITY_aboveUnderPertThreshold) {
            id.append("_PlurThr=").append(m_PluralityVotersThreshold);
        }

        if (aggregationVoters == AGGREGATION_LOWEST_WEIGHT_VOTER) {
            id.append("_MinWeigThr=").append(m_WeightedVoteThreshold);
        }

        id.append("_ConLearn=").append(getClosureStrategyName());

        if (closureStrategy >= CONCEPT_LEARNING_ABOVE_aboveUnderPertThreshold_ALL &&
            closureStrategy <= 13) {
            id.append("_Thr=").append(m_aboveUnderPertThreshold);
            id.append("_ThrTim=").append(m_beforeAfterDisc == 1 ? "BEF disc" : "AFT disc");
            id.append("_ThrCalc=").append(autoPertThresholdComb ? "Auto" : "Manual");
        }

        id.append("_Measure=").append(pertinenceMeasureToString(PertinenceMeasureMethod));

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
        
        switch (aggregationVoters) {
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
                
            case AGGREGATION_MIN_PLURALITY_aboveUnderPertThreshold:
                result = classifier.classify_Instance_nom_VotePlurMin(
                    filteredInst, m_classifierNC, nb_class, m_PluralityVotersThreshold, m_Debug);
                break;
                
            case AGGREGATION_LOWEST_WEIGHT_VOTER:
                result = classifier.classify_Instance_nom_VoteLowPondVoter(
                    filteredInst, m_classifierNC, nb_class, m_WeightedVoteThreshold, m_Debug);
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

    public double getm_Threshold(double[] pertinenceMeasures) {
        if (autoPertThresholdComb) {
            double sum = 0.0;
            for (double value : pertinenceMeasures) {
                sum += value;
            }
            m_aboveUnderPertThreshold = sum / pertinenceMeasures.length;
        }
        return m_aboveUnderPertThreshold;
    }

    /** JavaBean getter/setter pour la GUI Weka */
    public boolean getDebug() {
        return m_Debug;
    }

    public void setDebug(boolean debug) {
        this.m_Debug = debug;
    }

    public String debugTipText() {
        return "Enable debug output (prints diagnostic information)";
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
        float v,p;
        
        
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
     
        RuleExtractor extractor = new RuleExtractor(m_Debug, PertinenceMeasureMethod, policySelectionOccurence, closureStrategy,m_FilteredInstances);
       
        if (m_beforeAfterDisc == aboveUnderPertThreshold_BEFORE_DISCRETIZATION) {
        	m_aboveUnderPertThreshold=getm_Threshold(extractor.computePertinenceMeasures(data));
        	
        }
        else if ( m_beforeAfterDisc == aboveUnderPertThreshold_AFTER_DISCRETIZATION) {
        	m_aboveUnderPertThreshold=getm_Threshold(extractor.computePertinenceMeasures(m_FilteredInstances));

         }
    
        if (this.getfullAttribCombinations()==true) {
        	switch (closureStrategy) {
            case CONCEPT_LEARNING_ALL_ALL:
                if (m_Debug) {
                    System.out.println("=== ALL ATTRIBUTES - ALL VALUES - fullAttribCombinations MODE ACTIVE ===");
                    System.out.println("Extraction using all attributes with all values");
                    System.out.println("Used Measure: " + pertinenceMeasureToString(PertinenceMeasureMethod));
                }
                m_classifierNC = extractor.extractAllAttributesAllValuesfullAttribCombinations(m_FilteredInstances);
               
                break;
                
            case CONCEPT_LEARNING_ALL_MAJORITY:
                if (m_Debug) {
                    System.out.println("=== ALL ATTRIBUTES - MAJORITY VALUE - fullAttribCombinations MODE ACTIVE ===");
                    System.out.println("Extraction using all attributes with majority values");
                    System.out.println("Used Measure: " + pertinenceMeasureToString(PertinenceMeasureMethod));
                }
                m_classifierNC = extractor.extractAllAttributesMajorityValuefullAttribCombinations(m_FilteredInstances);
                if (m_Debug) {
                    printRuleStatistics();
                }
                break;
                
            case CONCEPT_LEARNING_PERTINENT_ALL:
                if (m_Debug) {
                    System.out.println("=== PERTINENT ATTRIBUTE - ALL VALUES - fullAttribCombinations MODE ACTIVE ==="); 
                    System.out.println("Extraction using only pertinent attribute with all values");
                    System.out.println("Used Measure: " + pertinenceMeasureToString(PertinenceMeasureMethod));
                    extractor.printPertinentAttribute(m_FilteredInstances);

                }
                m_classifierNC = extractor.extractPertinentAttributeAllValues(m_FilteredInstances);
                if (m_Debug) {
                    printRuleStatistics();
                }

                break;
                
            case CONCEPT_LEARNING_PERTINENT_MAJORITY:
                if (m_Debug) {
                    System.out.println("=== PERTINENT ATTRIBUTE - MAJORITY VALUE - fullAttribCombinations MODE ACTIVE ===");
                    System.out.println("Extraction using only pertinent attribute with majority value");
                    System.out.println("Used Measure: " + pertinenceMeasureToString(PertinenceMeasureMethod));
                    extractor.printPertinentAttribute(m_FilteredInstances);

                } 
                m_classifierNC = extractor.extractPertinentAttributeMajorityValue(m_FilteredInstances); 
                if (m_Debug) {
                    printRuleStatistics();
                }

                break;
                
            case CONCEPT_LEARNING_ABOVE_aboveUnderPertThreshold_ALL:
                if (m_Debug) {
                    System.out.println("=== ABOVE aboveUnderPertThreshold - ALL VALUES - fullAttribCombinations MODE ACTIVE ===");
                    if(autoPertThresholdComb) {
                        System.out.println("===Auto Threshold Mode Active=== ");

                    }
                    if ( m_beforeAfterDisc == aboveUnderPertThreshold_BEFORE_DISCRETIZATION) {
                    	setAboveUnderPertThreshold(getm_Threshold(extractor.computePertinenceMeasures(data)));
                    	System.out.println("Threshold timing::aboveUnderPertThreshold_BEFORE_DISCRETIZATION");

                    }
                    else if (m_beforeAfterDisc == aboveUnderPertThreshold_AFTER_DISCRETIZATION) {
                    	setAboveUnderPertThreshold(getm_Threshold(extractor.computePertinenceMeasures(m_FilteredInstances)));
                    	System.out.println("Threshold timing:: aboveUnderPertThreshold_AFTER_DISCRETIZATION");

                    }
                    System.out.printf("Extraction with (Threshold=%.4f) " ,m_aboveUnderPertThreshold);
                    System.out.println("Used Measure: " + pertinenceMeasureToString(PertinenceMeasureMethod));
                    extractor.printTopPercentageAttributes(m_FilteredInstances, m_aboveUnderPertThreshold);

                }
                m_classifierNC = extractor.extractAttributeAboveThresholdAllValuesfullAttribCombinations(m_FilteredInstances,m_aboveUnderPertThreshold);
                if (m_Debug) {
                    printRuleStatistics();
                }

                break;
                 
            case CONCEPT_LEARNING_ABOVE_aboveUnderPertThreshold_MAJORITY:
                if (m_Debug) {
                    System.out.println("=== ABOVE aboveUnderPertThreshold - MAJORITY VALUES - fullAttribCombinations MODE ACTIVE ===");
                    if(autoPertThresholdComb) {
                        System.out.println("===Auto Threshold Mode Active=== ");

                    }
                    if ( m_beforeAfterDisc == aboveUnderPertThreshold_BEFORE_DISCRETIZATION) {
                    	System.out.println("Threshold timing::aboveUnderPertThreshold_BEFORE_DISCRETIZATION");

                    }
                    else if (m_beforeAfterDisc == aboveUnderPertThreshold_AFTER_DISCRETIZATION) {
                    	System.out.println("Threshold timing:: aboveUnderPertThreshold_AFTER_DISCRETIZATION");

                    }
                    
                    System.out.printf("Extraction with(Threshold=%.4f) " ,m_aboveUnderPertThreshold);
                    System.out.println("Used Measure: " + pertinenceMeasureToString(PertinenceMeasureMethod));
                    extractor.printTopPercentageAttributes(m_FilteredInstances, m_aboveUnderPertThreshold);

                }
                m_classifierNC = extractor.extractAttributeAboveThresholdMajorityValuesfullAttribCombinations(m_FilteredInstances,m_aboveUnderPertThreshold);
                if (m_Debug) {
                    printRuleStatistics();
                }

                break;
                
            case CONCEPT_LEARNING_UNDER_aboveUnderPertThreshold_ALL:
                if (m_Debug) {
                    System.out.println("=== UNDER aboveUnderPertThreshold - ALL VALUES - fullAttribCombinations MODE ACTIVE ===");
                    if(autoPertThresholdComb) {
                        System.out.println("===Auto Threshold Mode Active=== ");

                    }
                    if ( m_beforeAfterDisc == aboveUnderPertThreshold_BEFORE_DISCRETIZATION) {
                    	System.out.println("Threshold timing::aboveUnderPertThreshold_BEFORE_DISCRETIZATION");

                    }
                    else if (m_beforeAfterDisc == aboveUnderPertThreshold_AFTER_DISCRETIZATION) {
                    	System.out.println("Threshold timing:: aboveUnderPertThreshold_AFTER_DISCRETIZATION");

                    }
                    System.out.printf("Extraction with (Threshold=%.4f) " ,m_aboveUnderPertThreshold);
                    System.out.println("Used Measure: " + pertinenceMeasureToString(PertinenceMeasureMethod));
                    extractor.printAttributesUnderThreshold(m_FilteredInstances, m_aboveUnderPertThreshold);

                }
                m_classifierNC = extractor.extractAttributeUnderThresholdAllValuesfullAttribCombinations
                		(m_FilteredInstances,m_aboveUnderPertThreshold);
                if (m_Debug) {
                    printRuleStatistics();
                }

                break;
                
            case CONCEPT_LEARNING_UNDER_aboveUnderPertThreshold_MAJORITY:
                if (m_Debug) {
                    System.out.println("=== UNDER aboveUnderPertThreshold - MAJORITY VALUES - fullAttribCombinations MODE ACTIVE ===");
                    if(autoPertThresholdComb) {
                        System.out.println("===Auto Threshold Mode Active=== ");

                    }
                    if ( m_beforeAfterDisc == aboveUnderPertThreshold_BEFORE_DISCRETIZATION) {
                    	System.out.println("Threshold timing::aboveUnderPertThreshold_BEFORE_DISCRETIZATION");

                    }
                    else if (m_beforeAfterDisc == aboveUnderPertThreshold_AFTER_DISCRETIZATION) {
                    	System.out.println("Threshold timing:: aboveUnderPertThreshold_AFTER_DISCRETIZATION");

                    }
                    System.out.printf("Extraction with(Threshold=%.4f) " ,m_aboveUnderPertThreshold);
                    System.out.println("Used Measure: " + pertinenceMeasureToString(PertinenceMeasureMethod));
                    extractor.printAttributesUnderThreshold(m_FilteredInstances, m_aboveUnderPertThreshold);

                }
                m_classifierNC = extractor.extractAttributeUnderThresholdAllValuesfullAttribCombinations(m_FilteredInstances,m_aboveUnderPertThreshold);
                if (m_Debug) {
                    printRuleStatistics();
                }
                break;
            case N_ATTRIBUTES_COMBINATION_ALL_VALUES:
                if (m_Debug) {
                    System.out.println("=== N Attribute - ALL VALUES - fullAttribCombinations MODE ACTIVE ===");
                }

                n = getAttributesCombinSize( );
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
                    System.out.println("=== N Attribute - MAJORITY VALUES - fullAttribCombinations MODE ACTIVE ===");
                }

                n = getAttributesCombinSize( );
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
                
            case PERCENT_ATTRIBUTES_COMBINATION_ALL_VALUES:
                if (m_Debug) {
                    System.out.println("=== Percent Attribute - ALL VALUES - fullAttribCombinations MODE ACTIVE ===");
                }

                p= this.getAttributesCombinPercent();
              
                m_classifierNC = extractor.extractPercentAttributesAllValues(m_FilteredInstances, p);
                if (m_Debug) {
                    printRuleStatistics();
                }
                break;
                
            case PERCENT_ATTRIBUTES_COMBINATION_ALL_VALUES_WITH_VAR:
                if (m_Debug) {
                    System.out.println("=== Percent Attribute WITH VARIANCE - ALL VALUES - fullAttribCombinations MODE ACTIVE ===");
                }

                p= this.getAttributesCombinPercent();
                v= this.getVarianceAttriCombPercent();
              
                m_classifierNC = extractor.extractPercentAttributesAllValuesWithVar(m_FilteredInstances, p,v);
                if (m_Debug) {
                    printRuleStatistics();
                }
                break;
                

        }
        }
        else if(this.getfullAttribCombinations()==false) {
        	switch (closureStrategy) {
            case CONCEPT_LEARNING_ALL_ALL:
                if (m_Debug) {
                    System.out.println("=== ALL ATTRIBUTES - ALL VALUES MODE ACTIVE ===");
                    System.out.println("Extraction using all attributes with all values");
                    System.out.println("Used Measure: " + pertinenceMeasureToString(PertinenceMeasureMethod));
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
                    System.out.println("Used Measure: " + pertinenceMeasureToString(PertinenceMeasureMethod));
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
                    System.out.println("Used Measure: " + pertinenceMeasureToString(PertinenceMeasureMethod));
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
                    System.out.println("Used Measure: " + pertinenceMeasureToString(PertinenceMeasureMethod));
                    extractor.printPertinentAttribute(m_FilteredInstances);

                } 
                m_classifierNC = extractor.extractPertinentAttributeMajorityValue(m_FilteredInstances); 
                if (m_Debug) {
                    printRuleStatistics();
                }

                break;
                
            case CONCEPT_LEARNING_ABOVE_aboveUnderPertThreshold_ALL:
                if (m_Debug) {
                    System.out.println("=== ABOVE aboveUnderPertThreshold - ALL VALUES MODE ACTIVE ===");
                    if(autoPertThresholdComb) {
                        System.out.println("===Auto Threshold Mode Active=== ");

                    }
                    if ( m_beforeAfterDisc == aboveUnderPertThreshold_BEFORE_DISCRETIZATION) {
                    	setAboveUnderPertThreshold(getm_Threshold(extractor.computePertinenceMeasures(data)));
                    	System.out.println("Threshold timing::aboveUnderPertThreshold_BEFORE_DISCRETIZATION");

                    }
                    else if (m_beforeAfterDisc == aboveUnderPertThreshold_AFTER_DISCRETIZATION) {
                    	setAboveUnderPertThreshold(getm_Threshold(extractor.computePertinenceMeasures(m_FilteredInstances)));
                    	System.out.println("Threshold timing:: aboveUnderPertThreshold_AFTER_DISCRETIZATION");

                    }
                    System.out.printf("Extraction with (Threshold=%.4f) " ,m_aboveUnderPertThreshold);
                    System.out.println("Used Measure: " + pertinenceMeasureToString(PertinenceMeasureMethod));
                    extractor.printTopPercentageAttributes(m_FilteredInstances, m_aboveUnderPertThreshold);

                }
                m_classifierNC = extractor.extractAttributeAboveThresholdAllValues(m_FilteredInstances,m_aboveUnderPertThreshold);
                if (m_Debug) {
                    printRuleStatistics();
                }

                break;
                
            case CONCEPT_LEARNING_ABOVE_aboveUnderPertThreshold_MAJORITY:
                if (m_Debug) {
                    System.out.println("=== ABOVE aboveUnderPertThreshold - MAJORITY VALUES MODE ACTIVE ===");
                    if(autoPertThresholdComb) {
                        System.out.println("===Auto Threshold Mode Active=== ");

                    }
                    if ( m_beforeAfterDisc == aboveUnderPertThreshold_BEFORE_DISCRETIZATION) {
                    	System.out.println("Threshold timing::aboveUnderPertThreshold_BEFORE_DISCRETIZATION");

                    }
                    else if (m_beforeAfterDisc == aboveUnderPertThreshold_AFTER_DISCRETIZATION) {
                    	System.out.println("Threshold timing:: aboveUnderPertThreshold_AFTER_DISCRETIZATION");

                    }
                    
                    System.out.printf("Extraction with(Threshold=%.4f) " ,m_aboveUnderPertThreshold);
                    System.out.println("Used Measure: " + pertinenceMeasureToString(PertinenceMeasureMethod));
                    extractor.printTopPercentageAttributes(m_FilteredInstances, m_aboveUnderPertThreshold);

                }
                m_classifierNC = extractor.extractAttributeAboveThresholdMajorityValues(m_FilteredInstances,m_aboveUnderPertThreshold);
                if (m_Debug) {
                    printRuleStatistics();
                }

                break;
                
            case CONCEPT_LEARNING_UNDER_aboveUnderPertThreshold_ALL:
                if (m_Debug) {
                    System.out.println("=== UNDER aboveUnderPertThreshold - ALL VALUES MODE ACTIVE ===");
                    if(autoPertThresholdComb) {
                        System.out.println("===Auto Threshold Mode Active=== ");

                    }
                    if ( m_beforeAfterDisc == aboveUnderPertThreshold_BEFORE_DISCRETIZATION) {
                    	System.out.println("Threshold timing::aboveUnderPertThreshold_BEFORE_DISCRETIZATION");

                    }
                    else if (m_beforeAfterDisc == aboveUnderPertThreshold_AFTER_DISCRETIZATION) {
                    	System.out.println("Threshold timing:: aboveUnderPertThreshold_AFTER_DISCRETIZATION");

                    }
                    System.out.printf("Extraction with (Threshold=%.4f) " ,m_aboveUnderPertThreshold);
                    System.out.println("Used Measure: " + pertinenceMeasureToString(PertinenceMeasureMethod));
                    extractor.printAttributesUnderThreshold(m_FilteredInstances, m_aboveUnderPertThreshold);

                }
                m_classifierNC = extractor.extractAttributeUnderThresholdAllValues
                		(m_FilteredInstances,m_aboveUnderPertThreshold);
                if (m_Debug) {
                    printRuleStatistics();
                }

                break;
                
            case CONCEPT_LEARNING_UNDER_aboveUnderPertThreshold_MAJORITY:
                if (m_Debug) {
                    System.out.println("=== UNDER aboveUnderPertThreshold - MAJORITY VALUES MODE ACTIVE ===");
                    if(autoPertThresholdComb) {
                        System.out.println("===Auto Threshold Mode Active=== ");

                    }
                    if ( m_beforeAfterDisc == aboveUnderPertThreshold_BEFORE_DISCRETIZATION) {
                    	System.out.println("Threshold timing::aboveUnderPertThreshold_BEFORE_DISCRETIZATION");

                    }
                    else if (m_beforeAfterDisc == aboveUnderPertThreshold_AFTER_DISCRETIZATION) {
                    	System.out.println("Threshold timing:: aboveUnderPertThreshold_AFTER_DISCRETIZATION");

                    }
                    System.out.printf("Extraction with(Threshold=%.4f) " ,m_aboveUnderPertThreshold);
                    System.out.println("Used Measure: " + pertinenceMeasureToString(PertinenceMeasureMethod));
                    extractor.printAttributesUnderThreshold(m_FilteredInstances, m_aboveUnderPertThreshold);

                }
                m_classifierNC = extractor.extractAttributeUnderThresholdMajorityValues(m_FilteredInstances,m_aboveUnderPertThreshold);
                if (m_Debug) {
                    printRuleStatistics();
                }

                break;
        }
        	
        }

        List<FormalConcept> concepts=extractor.m_GaloisConnection.getConceptList();
        
        this.conceptsListStr=extractor.m_GaloisConnection.getConceptLevelsAsString(concepts);
        vectorialConceptList=extractor.m_GaloisConnection.vectorialConceptList;
    }
    
  
    
    public static String pertinenceMeasureToString(int measureType) {
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
        if (m_classifierNC == null || m_classifierNC.isEmpty()) {
            System.out.println("No rules generated.");
            return;
        }

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
    
    @Override
    public String toString() {

        // Vérifie si on est en mode expérimentation
        if (isInExperimentMode()) {
            return super.toString();
        }

        StringBuilder sb = new StringBuilder();
        SimpleDateFormat sdf = new SimpleDateFormat("EEE yyyy-MM-dd HH:mm:ss.SSS");
        String currentTime = sdf.format(new Date());

        // En-tête
        sb.append("=== CNC Classifier Output ===\n");
        sb.append("Generated at: ").append(currentTime).append("\n\n");

        // Configuration du classifieur
        sb.append("=== Configuration ===\n");
        sb.append(String.format("%-30s: %s\n", "Concept Learning Strategy", getClosureStrategyName()));
        sb.append(String.format("%-30s: %s\n", "Feature Measure", pertinenceMeasureToString(PertinenceMeasureMethod)));
        sb.append(String.format("%-30s: %s\n", "Auto Pertinence Threshold", autoPertThresholdCombToString()));
        sb.append(String.format("%-30s: %s\n", "Threshold Calculation", m_beforeAfterDiscToString()));
        sb.append(String.format("%-30s: %.4f\n", "Above/Under Pert Threshold", m_aboveUnderPertThreshold));
        sb.append(String.format("%-30s: %s\n", "Policy Selection ", getPolicySelectionOccurenceName()));
        sb.append(String.format("%-30s: %s\n", "Aggregation Voters ", getaggregationVotersName()));
        sb.append(String.format("%-30s: %d\n", "Top-K Voters", m_TopKVoters));
        sb.append(String.format("%-30s: %.4f\n", "Plurality Threshold", m_PluralityVotersThreshold));
        sb.append(String.format("%-30s: %.4f\n", "Weight Threshold", m_WeightedVoteThreshold));
        sb.append(String.format("%-30s: %d\n", "Attributes Combination Size", attributesCombinSize));
        sb.append(String.format("%-30s: %.2f\n", "Attributes Combination Percent", attributesCombinPercent));
        sb.append(String.format("%-30s: %.2f\n", "Variance Attributes Comb %", getVarianceAttriCombPercent()));
        sb.append(String.format("%-30s: %s\n", "Full Attributes Combinations", getfullAttribCombinations()));
        sb.append(String.format("%-30s: %s\n", "Filter Specification", getFilterSpec()));
        sb.append(String.format("%-30s: %b\n", "Debug Enabled", m_Debug));

        sb.append("\n");

        // Concepts
        sb.append("=== Concepts ===\n");
        sb.append(conceptsListStr).append("\n");

        // Liste vectorielle
        if (vectorialConceptList != null && !vectorialConceptList.isEmpty()) {
            sb.append("\n=== Vectorial Concept List ===\n");
            sb.append(vectorialConceptList).append("\n");
        }

        // Statistiques de performance
        if (m_classifierNC != null && !m_classifierNC.isEmpty()) {
            sb.append("=== Performance ===\n");
            sb.append(String.format("%-30s: %d\n", "Number of Rules Generated", m_classifierNC.size()));
            sb.append("\n");

            sb.append("=== Generated Rules ===\n");
            for (int i = 0; i < m_classifierNC.size(); i++) {
                sb.append(String.format("Rule %d:  ", i + 1));
                sb.append(m_classifierNC.get(i).affich_nom_rule(true));
                sb.append("\n");
            }
        } else if (m_ZeroR != null) {
            sb.append("Using ZeroR model as fallback\n");
        } else {
            sb.append("No model generated\n");
        }

        // Debug
        if (m_Debug) {
            sb.append("\n=== Debug Information ===\n");
            sb.append("Log console available\n");
            sb.append("Last execution: ").append(currentTime).append("\n");
        }

        return sb.toString();
    }



    private String getPolicySelectionOccurenceName() {
        switch (policySelectionOccurence) {
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