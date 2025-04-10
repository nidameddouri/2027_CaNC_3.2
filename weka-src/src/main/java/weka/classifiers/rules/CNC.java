package weka.classifiers.rules;

import java.util.ArrayList;
import java.util.Enumeration;
import java.util.Vector;

import weka.attributeSelection.*;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.classifiers.Sourcable;
import weka.core.*;
import weka.core.Capabilities.Capability;
import weka.classifiers.UpdateableClassifier;
import weka.classifiers.fca.Classification_Rule;
import weka.classifiers.fca.Classify_Instance;
import weka.classifiers.fca._GaloisConnection;
import weka.classifiers.fca.RuleExtractor;
import weka.core.TechnicalInformation.Field;
import weka.core.TechnicalInformation.Type;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.ReplaceMissingValues;

import java.text.SimpleDateFormat;
import java.util.Calendar;

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

    // ConceptLearning is wheteher to use pertinent attribute or all attributes to compute the closure
    public static final int CONCEPT_LEARNING_PERTINENT = 1;
    public static final int CONCEPT_LEARNING_ALL = 2;
    
    private int ConceptLearningStrategy = CONCEPT_LEARNING_PERTINENT; // Default value : Pertinent
    
    public static final Tag[] TAGS_ConceptLearning = {
        new Tag(CONCEPT_LEARNING_PERTINENT, "Closure of the pertinent attribute"),
        new Tag(CONCEPT_LEARNING_ALL, "Closure of ALL attributes"),
    };
    
    public SelectedTag getConceptLearning() {
        return new SelectedTag(ConceptLearningStrategy, TAGS_ConceptLearning);
    }
    
    public void setConceptLearning(SelectedTag value) {
        if (value.getTags() == TAGS_ConceptLearning) {
            this.ConceptLearningStrategy = value.getSelectedTag().getID();
        }
    }

    // ValueStrategy is the strategy used to compute the value of the attribute
    
    public static final int VALUE_STRATEGY_BEST = 1;
    public static final int VALUE_STRATEGY_ALL = 2;
    
    private int ValueStrategy = VALUE_STRATEGY_BEST; // Default value : Best Value
    
    public static final Tag[] TAGS_ValueStrategy = {
        new Tag(VALUE_STRATEGY_BEST, "Best Value"),
        new Tag(VALUE_STRATEGY_ALL, "All Values"),
    };
    
    public SelectedTag getValueStrategy() {
        return new SelectedTag(ValueStrategy, TAGS_ValueStrategy);
    }
    
    public void setValueStrategy(SelectedTag value) {
        if (value.getTags() == TAGS_ValueStrategy) {
            this.ValueStrategy = value.getSelectedTag().getID();
        }
    }

    // Policy_Selection is the policy used to select the attribute in case of equality
    
    public static final int POLICY_FIRST = 1;
    public static final int POLICY_LAST = 2;
    
    private int Policy_Selection = POLICY_LAST; // Default value : Last
    
    public static final Tag[] TAGS_Policy_Selection = {
        new Tag(POLICY_FIRST, "First attribute in case of equality"),
        new Tag(POLICY_LAST, "Last attribute in case of equality"),
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

    // GaloisConnection is whether to use Galois connection or not
    // Note : this function isn't tested at the moment

    public static final boolean GALOIS_DISABLED = false;
    public static final boolean GALOIS_ENABLED = true;

    private boolean UseGaloisConnection = GALOIS_DISABLED; // Default value : Disabled

    public static final Tag[] TAGS_GaloisConnection = {
        new Tag(0, "Disabled"),
        new Tag(1, "Enabled"),
    };

    public SelectedTag get_GaloisConnection() {
        return new SelectedTag(UseGaloisConnection ? 1 : 0, TAGS_GaloisConnection);
    }

    public void set_GaloisConnection(SelectedTag value) {
        if (value.getTags() == TAGS_GaloisConnection) {
            UseGaloisConnection = (value.getSelectedTag().getID() == 1);
        }
    }

    // GaloisStrategy is the strategy used to compute the Galois connection

    public static final int GALOIS_STRATEGY_PERTINENT_MAJORITY = 1;
    public static final int GALOIS_STRATEGY_PERTINENT_ALL = 2;
    public static final int GALOIS_STRATEGY_ALL_MAJORITY = 3;
    public static final int GALOIS_STRATEGY_ALL_ALL = 4;

    private int GaloisStrategy = GALOIS_STRATEGY_PERTINENT_MAJORITY; // Default value : Pertinent Attribute & Majority Value

    public static final Tag[] TAGS_GaloisStrategy = {
        new Tag(GALOIS_STRATEGY_PERTINENT_MAJORITY, "Pertinent Attribute & Majority Value"),
        new Tag(GALOIS_STRATEGY_PERTINENT_ALL, "Pertinent Attribute & All Values"),
        new Tag(GALOIS_STRATEGY_ALL_MAJORITY, "All Attributes & Majority Value"),
        new Tag(GALOIS_STRATEGY_ALL_ALL, "All Attributes & All Values"),
    };

    public SelectedTag get_GaloisStrategy() {
        return new SelectedTag(GaloisStrategy, TAGS_GaloisStrategy);
    }

    public void set_GaloisStrategy(SelectedTag value) {
        if (value.getTags() == TAGS_GaloisStrategy) {
            GaloisStrategy = value.getSelectedTag().getID();
        }
    }
    
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
             + getTechnicalInformation().toString() + "\n\nThis code was rewritten and improved by Francoise Lapetite & Younes Kara Zaitri\n";
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
    

    // listOptions() returns a list of the options available for this classifier, it is important in order to add a parameter
    public Enumeration<Option> listOptions() {
        Vector<Option> newVector = new Vector<>(10);
        
        newVector.addElement(new Option("\tUse pertinent attribute concept learning.", "pertinent", 0, "-pertinent"));
        newVector.addElement(new Option("\tUse all attributes concept learning.", "all", 0, "-all"));
        newVector.addElement(new Option("\tUse best value strategy.", "bestValue", 0, "-bestValue"));
        newVector.addElement(new Option("\tUse all values strategy.", "allValues", 0, "-allValues"));
        
        newVector.addElement(new Option("\tEnable Galois connection.", "_galois", 0, "-_galois"));
        newVector.addElement(new Option("\tGalois strategy (1=PERTINENT_MAJORITY, 2=PERTINENT_ALL, 3=ALL_MAJORITY, 4=ALL_ALL).", 
                                       "galoisStrategy", 1, "-galoisStrategy <num>"));
        
        newVector.addElement(new Option("\tFeature selection measure (entropy, infogain, gainratio)", 
                                       "FeatureMeasure", 1, "-FeatureMeasure <string>"));
        
        newVector.addElement(new Option(
            "\t Full class name of filter to use, followed\n"
            + "\t by filter options.\n"
            + "\t eg: \"weka.filters.supervised.attribute.Remove -V -R 1,2\"",
            "F", 1, "-F <filter specification>"));
        
        Enumeration<Option> enu = super.listOptions();
        while (enu.hasMoreElements()) {
            newVector.addElement(enu.nextElement());
        }
        
        return newVector.elements();
    }


    // setOptions() sets the options for the classifier
    public void setOptions(String[] options) throws Exception {
        boolean runString;
        
        String featMeasure = Utils.getOption("FeatureMeasure", options);
        if (featMeasure.length() != 0) {
            if (featMeasure.equalsIgnoreCase("entropy"))
                FeatureMeasureMethod = FEAT_ENTROPY;
            else if (featMeasure.equalsIgnoreCase("infogain"))
                FeatureMeasureMethod = FEAT_INFOGAIN;
            else if (featMeasure.equalsIgnoreCase("gainratio"))
                FeatureMeasureMethod = FEAT_GAINRATIO;
        }
        
        runString = Utils.getFlag("pertinent", options);
        if (runString)
            ConceptLearningStrategy = CONCEPT_LEARNING_PERTINENT;
        
        runString = Utils.getFlag("all", options);
        if (runString)
            ConceptLearningStrategy = CONCEPT_LEARNING_ALL;
        
        runString = Utils.getFlag("bestValue", options);
        if (runString)
            ValueStrategy = VALUE_STRATEGY_BEST;
        
        runString = Utils.getFlag("allValues", options);
        if (runString)
            ValueStrategy = VALUE_STRATEGY_ALL;
        
        runString = Utils.getFlag("_galois", options);
        if (runString)
            UseGaloisConnection = GALOIS_ENABLED;
        
        String galoisStrategyStr = Utils.getOption("_galoisStrategy", options);
        if (galoisStrategyStr.length() != 0) {
            try {
                GaloisStrategy = Integer.parseInt(galoisStrategyStr);
                if (GaloisStrategy < 1 || GaloisStrategy > 4)
                    GaloisStrategy = GALOIS_STRATEGY_PERTINENT_MAJORITY;
            } catch (NumberFormatException e) {
                GaloisStrategy = GALOIS_STRATEGY_PERTINENT_MAJORITY;
            }
        }
        
        runString = Utils.getFlag("policyFirst", options);
        if (runString)
            Policy_Selection = POLICY_FIRST;
        
        runString = Utils.getFlag("policyLast", options);
        if (runString)
            Policy_Selection = POLICY_LAST;
        
        m_Debug = Utils.getFlag("debug", options);
        
        String filterString = Utils.getOption('F', options);
        if (filterString.length() > 0) {
            String[] filterSpec = Utils.splitOptions(filterString);
            if (filterSpec.length == 0)
                throw new IllegalArgumentException("Invalid filter specification string");
            String filterName = filterSpec[0];
            filterSpec[0] = "";
            setFilter((Filter) Utils.forName(Filter.class, filterName, filterSpec));
        } else {
            setFilter(new weka.filters.supervised.attribute.Discretize());
        }
        
        super.setOptions(options);
    }


    // getOptions() returns the options for the classifier
    public String[] getOptions() {
        Vector<String> result = new Vector<>();
        
        switch (ConceptLearningStrategy) {
            case CONCEPT_LEARNING_PERTINENT:
                result.add("-pertinent");
                break;
            case CONCEPT_LEARNING_ALL:
                result.add("-all");
                break;
        }
        
        switch (ValueStrategy) {
            case VALUE_STRATEGY_BEST:
                result.add("-bestValue");
                break;
            case VALUE_STRATEGY_ALL:
                result.add("-allValues");
                break;
        }
        
        if (UseGaloisConnection) {
            result.add("-_galois");
            result.add("-_galoisStrategy");
            result.add("" + GaloisStrategy);
        }
        
        switch (Policy_Selection) {
            case POLICY_FIRST:
                result.add("-policyFirst");
                break;
            case POLICY_LAST:
                result.add("-policyLast");
                break;
        }
        
        if (m_Debug) {
            result.add("-debug");
        }
        
        result.add("-F");
        result.add(getFilterSpec());
        
        switch (FeatureMeasureMethod) {
            case FEAT_ENTROPY:
                result.add("-FeatureMeasure");
                result.add("entropy");
                break;
            case FEAT_INFOGAIN:
                result.add("-FeatureMeasure");
                result.add("infogain");
                break;
            case FEAT_GAINRATIO:
                result.add("-FeatureMeasure");
                result.add("gainratio");
                break;
        }
        
        String[] superOptions = super.getOptions();
        for (String option : superOptions) {
            result.add(option);
        }
        
        return result.toArray(new String[0]);
    }

    // classifyInstance() classifies a given instance, it is mandatory to implement this function
    @Override
    public double classifyInstance(Instance inst) throws Exception {
        if (m_Filter.numPendingOutput() > 0)
            throw new Exception("Filter output queue not empty!");
        
        if (!m_Filter.input(inst))
            throw new Exception("Filter didn't make the test instance immediately available!");
        
        m_Filter.batchFinished();
        Instance newInstance = m_Filter.output();
        
        m_Missing.input(inst);
        m_Missing.batchFinished();
        inst = m_Missing.output();
        
        double result = (double) -1.0;
        Classify_Instance listRules = new Classify_Instance();
        
        if (ConceptLearningStrategy == CONCEPT_LEARNING_PERTINENT && ValueStrategy == VALUE_STRATEGY_BEST) {
            result = (double) listRules.classify_Instance_nom(newInstance, m_classifierNC.get(0));
        } else {
            result = (double) listRules.classify_Instance_nom_VoteLowPondVoter(
                newInstance, m_classifierNC, newInstance.numClasses(), m_LowPondVoter);
        }
        
        if (m_Debug) {
            System.out.println("- Instance: " + newInstance.toString());

            if ((int) result == -1) {
                System.out.println("\tDecision: -1 (Class: undetermined - no applicable rule)");
            } else if ((int) result >= 0 && (int) result < newInstance.classAttribute().numValues()) {
                String className = newInstance.classAttribute().value((int) result);
                System.out.println("\tDecision: " + (int) result + " (Class: " + className + ")");
            } else {
                System.out.println("\tDecision: " + (int) result + " (Class: unknown - index out of bounds)");
            }

            System.out.println("\n");
        }
        
        if (result == -1.0)
            return Utils.missingValue();
        
        return result;
    }

    // buildClassifier() builds the classifier using the training data, it is mandatory to implement this function

    public void buildClassifier(Instances data) throws Exception {
        getCapabilities().testWithFail(data);
        
        m_FilteredInstances = new Instances(data);
        m_FilteredInstances.deleteWithMissingClass();
        
        if (m_FilteredInstances.numInstances() == 0)
            throw new Exception("No training instances left after removing instances with MissingClass!");
        
        m_Missing.setInputFormat(m_FilteredInstances);
        m_FilteredInstances = Filter.useFilter(m_FilteredInstances, m_Missing);
        
        if (m_FilteredInstances.numInstances() == 0)
            throw new Exception("No training instances left after removing instances with MissingValues!");
        
        m_Filter.setInputFormat(m_FilteredInstances);
        m_FilteredInstances = Filter.useFilter(m_FilteredInstances, m_Filter);
        
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
        
        if (UseGaloisConnection) {
            if (m_Debug) {
                System.out.println("Using Galois connection with strategy: " + GaloisStrategy);
            }
            
            _GaloisConnection galoisConnection = new _GaloisConnection(m_Debug);
            galoisConnection.buildConnection(m_FilteredInstances);
            
            m_classifierNC = galoisConnection.generateRulesWithStrategy(GaloisStrategy);
        } else {
            RuleExtractor extractor = new RuleExtractor(m_Debug, FeatureMeasureMethod, Policy_Selection, ConceptLearningStrategy);
            
            if (ConceptLearningStrategy == CONCEPT_LEARNING_PERTINENT) {
                if (ValueStrategy == VALUE_STRATEGY_BEST) {
                    m_classifierNC = extractor.ExtraireRegleFermNom(m_FilteredInstances, 1);
                } else {
                    m_classifierNC = extractor.ExtraireRegleFermNom(m_FilteredInstances, 3);
                }
            } else {
                ArrayList<Classification_Rule> tempRules = new ArrayList<>();
                
                for (int i = 0; i < m_FilteredInstances.numAttributes(); i++) {
                    if (i == m_FilteredInstances.classIndex()) continue;
                    
                    if (ValueStrategy == VALUE_STRATEGY_BEST) {
                        Attribute attr = m_FilteredInstances.attribute(i);
                        
                        int bestValueIdx = 0;
                        int bestSupport = 0;
                        
                        for (int v = 0; v < attr.numValues(); v++) {
                            int support = 0;
                            for (int j = 0; j < m_FilteredInstances.numInstances(); j++) {
                                if (m_FilteredInstances.instance(j).value(i) == v) {
                                    support++;
                                }
                            }
                            
                            if (support > bestSupport) {
                                bestSupport = support;
                                bestValueIdx = v;
                            }
                        }
                        
                        ArrayList<Integer> instanceIndices = new ArrayList<>();
                        for (int j = 0; j < m_FilteredInstances.numInstances(); j++) {
                            if (m_FilteredInstances.instance(j).value(i) == bestValueIdx) {
                                instanceIndices.add(j);
                            }
                        }
                        
                        if (!instanceIndices.isEmpty()) {
                            ArrayList<String> attributes = new ArrayList<>();
                            for (int a = 0; a < m_FilteredInstances.numAttributes() - 1; a++) {
                                attributes.add("-null-");
                            }
                            
                            int attrIdx = (i < m_FilteredInstances.classIndex()) ? i : i - 1;
                            attributes.set(attrIdx, attr.value(bestValueIdx));
                            
                            int[] classCounts = new int[m_FilteredInstances.numClasses()];
                            for (int idx : instanceIndices) {
                                classCounts[(int)m_FilteredInstances.instance(idx).classValue()]++;
                            }
                            
                            int majorityClass = 0;
                            for (int c = 0; c < classCounts.length; c++) {
                                if (classCounts[c] > classCounts[majorityClass]) {
                                    majorityClass = c;
                                }
                            }
                            
                            Classification_Rule rule = new Classification_Rule(
                                attributes,
                                majorityClass,
                                m_FilteredInstances.attribute(m_FilteredInstances.classIndex()).value(majorityClass)
                            );
                            
                            Classify_Instance classifier = new Classify_Instance();
                            double weight = 0;
                            for (int j = 0; j < m_FilteredInstances.numInstances(); j++) {
                                if (classifier.classify_Instance_nom(m_FilteredInstances.instance(j), rule) != -1.0) {
                                    weight++;
                                }
                            }
                            rule.setRule_Ponderation(weight / m_FilteredInstances.numInstances());
                            
                            tempRules.add(rule);
                        }
                    } else {
                        Attribute attr = m_FilteredInstances.attribute(i);
                        
                        for (int v = 0; v < attr.numValues(); v++) {
                            ArrayList<Integer> instanceIndices = new ArrayList<>();
                            for (int j = 0; j < m_FilteredInstances.numInstances(); j++) {
                                if (m_FilteredInstances.instance(j).value(i) == v) {
                                    instanceIndices.add(j);
                                }
                            }
                            
                            if (!instanceIndices.isEmpty()) {
                                ArrayList<String> attributes = new ArrayList<>();
                                for (int a = 0; a < m_FilteredInstances.numAttributes() - 1; a++) {
                                    attributes.add("-null-");
                                }
                                
                                int attrIdx = (i < m_FilteredInstances.classIndex()) ? i : i - 1;
                                attributes.set(attrIdx, attr.value(v));
                                
                                int[] classCounts = new int[m_FilteredInstances.numClasses()];
                                for (int idx : instanceIndices) {
                                    classCounts[(int)m_FilteredInstances.instance(idx).classValue()]++;
                                }
                                
                                int majorityClass = 0;
                                for (int c = 0; c < classCounts.length; c++) {
                                    if (classCounts[c] > classCounts[majorityClass]) {
                                        majorityClass = c;
                                    }
                                }
                                
                                Classification_Rule rule = new Classification_Rule(
                                    attributes,
                                    majorityClass,
                                    m_FilteredInstances.attribute(m_FilteredInstances.classIndex()).value(majorityClass)
                                );
                                
                                Classify_Instance classifier = new Classify_Instance();
                                double weight = 0;
                                for (int j = 0; j < m_FilteredInstances.numInstances(); j++) {
                                    if (classifier.classify_Instance_nom(m_FilteredInstances.instance(j), rule) != -1.0) {
                                        weight++;
                                    }
                                }
                                rule.setRule_Ponderation(weight / m_FilteredInstances.numInstances());
                                
                                tempRules.add(rule);
                            }
                        }
                    }
                }
                
                m_classifierNC = tempRules;
            }
        }
        
        if (m_Debug) {
            System.out.println("\n\n--- RULE GENERATION ---\n");
            
            for (int i = 0; i < m_classifierNC.size(); i++) {
                Classification_Rule rule = m_classifierNC.get(i);
                double coverageCount = 0;
                for (int j = 0; j < m_FilteredInstances.numInstances(); j++) {
                    try {
                        if (new Classify_Instance().classify_Instance_nom(m_FilteredInstances.get(j), rule) != -1.0)
                            coverageCount++;
                    } catch (Exception e) {
                        System.err.println("Error during rule evaluation: " + e.getMessage());
                    }
                }
                double poids = rule.getRule_Ponderation();
                System.out.printf("--- Rule #%d ---\n", i + 1);
                System.out.println(rule.affich_nom_rule(true));
                System.out.printf("\n");
                System.out.printf("--- RECAP ---\n");
                System.out.printf("Covers: %.0f / %d instances\n", coverageCount, m_FilteredInstances.numInstances());
                System.out.printf("Poids: %.2f%%\n", poids * 100);
                System.out.printf("-------------\n");
                System.out.printf("\n");

                System.out.println("\n\n--- CLASSIFICATION OF DIFFERENT CONTEXTS ---\n");
            }
        }
    }
    

    public String toString() {
        Calendar calendar = Calendar.getInstance();
        SimpleDateFormat sdf = new SimpleDateFormat("E MM/dd/yyyy HH:mm:ss.SSS");
        
        StringBuffer sb = new StringBuffer(sdf.format(calendar.getTime()) + "\t [Classifier all Nominal Concept Notes] \n");
        calendar = Calendar.getInstance();
        
        if ((m_CNC == 1) && (this.m_ZeroR == null)) {
            sb.append(sdf.format(calendar.getTime()) + "\t Running OK ... \n");
            sb.append(sdf.format(calendar.getTime()) + "\t The next display can be missing ... \n");
            for (int i = 0; i < m_classifierNC.size(); i++)
                sb.append(sdf.format(calendar.getTime()) + "\t Classifier " + (i + 1) + " : " 
                     + m_classifierNC.get(i).affich_nom_rule(true) + "\n");
            if (m_Debug)
                sb.append(sdf.format(calendar.getTime()) + " \t A Log Console is generated... \n");
            return sb.toString();
        } else
            return sdf.format(calendar.getTime()) + " \t m_ZeroR model \n";
    }
    

    // Those functions are mandatory because of the interface, but they are not used in this implementation
    @Override
    public String toSource(String className) throws Exception {
        return null;
    }
    
    @Override
    public void updateClassifier(Instance instance) throws Exception {
        System.out.println("/** updateClassifier(Instance instance) **/");
    }

    // Main function to run the classifier
    
    public static void main(String[] argv) {
        runClassifier(new CNC(), argv);
    }
}