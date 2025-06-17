package weka.classifiers.fca;


import weka.core.*;
import java.util.*;

public class TestClassificationRule {
    
    public static void main(String[] args) throws Exception {
        // 1. Création d'un dataset de test
        Instances testDataset = createTestDataset();
        
        // 2. Création de règles de test
        System.out.println("=== Création des règles de test ===");
        Classification_Rule rule1 = createTestRule1();
        Classification_Rule rule2 = createTestRule2(testDataset);
        Classification_Rule rule3 = createTestRule3();
        
        // 3. Tests des méthodes d'affichage
        System.out.println("\n=== Test des méthodes d'affichage ===");
        testDisplayMethods(rule1);
        
        // 4. Tests des méthodes de comparaison
        System.out.println("\n=== Test des méthodes de comparaison ===");
        testComparisonMethods(rule1, rule2, rule3);
        
        // 5. Test avec un vrai dataset
        System.out.println("\n=== Test avec création depuis un vrai dataset ===");
        testWithRealDataset(testDataset);
    }
    
    private static Instances createTestDataset() {
        // Création des attributs
        ArrayList<Attribute> attributes = new ArrayList<>();
        attributes.add(new Attribute("outlook", Arrays.asList("sunny", "overcast", "rainy")));
        attributes.add(new Attribute("temperature", Arrays.asList("hot", "mild", "cool")));
        attributes.add(new Attribute("humidity", Arrays.asList("high", "normal")));
        attributes.add(new Attribute("windy", Arrays.asList("TRUE", "FALSE")));
        attributes.add(new Attribute("play", Arrays.asList("yes", "no")));
        
        // Création du dataset
        Instances dataset = new Instances("weather", attributes, 0);
        dataset.setClassIndex(dataset.numAttributes() - 1);
        
        // Ajout de quelques instances
        addInstance(dataset, new double[]{0, 0, 0, 1, 0}); // sunny,hot,high,FALSE,yes
        addInstance(dataset, new double[]{0, 1, 0, 0, 1});  // sunny,mild,high,TRUE,no
        addInstance(dataset, new double[]{2, 2, 1, 1, 0});  // rainy,cool,normal,FALSE,yes
        
        return dataset;
    }
    
    private static void addInstance(Instances dataset, double[] values) {
        Instance instance = new DenseInstance(1.0, values);
        dataset.add(instance);
    }
    
    private static Classification_Rule createTestRule1() {
        System.out.println("Création de la règle 1 (manuelle)");
        Map<String, String> attributes = new HashMap<>();
        attributes.put("outlook", "sunny");
        attributes.put("temperature", "hot");
        
        Classification_Rule rule = new Classification_Rule(attributes, 0, "yes", 0.75);
        System.out.println("Règle créée: " + rule.affich_nom_rule(true));
        return rule;
    }
    
    private static Classification_Rule createTestRule2(Instances dataset) throws Exception {
        System.out.println("\nCréation de la règle 2 (à partir du dataset)");
        
        // Sélection des instances correspondant à humidity=normal
        ArrayList<Integer> instances = new ArrayList<>();
        for (int i = 0; i < dataset.numInstances(); i++) {
            if (dataset.instance(i).stringValue(2).equals("normal")) {
                instances.add(i);
            }
        }
        
        // Création de la fermeture
        ArrayList<String> closure = new ArrayList<>();
        for (int i = 0; i < dataset.numAttributes() - 1; i++) {
            closure.add(dataset.instance(instances.get(0)).stringValue(i));
        }
        
        // Conversion en Map
        Map<String, String> attrMap = new HashMap<>();
        for (int i = 0; i < closure.size(); i++) {
            attrMap.put(dataset.attribute(i).name(), closure.get(i));
        }
        
        Classification_Rule rule = Classification_Rule.createRule(dataset, instances, attrMap);
        System.out.println("Règle créée: " + rule.affich_nom_rule(true));
        return rule;
    }
    
    private static Classification_Rule createTestRule3() {
        System.out.println("\nCréation de la règle 3 (similaire à rule1)");
        Map<String, String> attributes = new HashMap<>();
        attributes.put("outlook", "sunny");
        attributes.put("temperature", "hot");
        
        Classification_Rule rule = new Classification_Rule(attributes, 0, "yes", 0.8);
        System.out.println("Règle créée: " + rule.affich_nom_rule(true));
        return rule;
    }
    
    private static void testDisplayMethods(Classification_Rule rule) {
        System.out.println("\nTest affich_nom_rule():");
        System.out.println(rule.affich_nom_rule(true));
        
        System.out.println("\nTest affich_csv_rule():");
        System.out.println(rule.affich_csv_rule(true));
        
        System.out.println("\nTest affich_text_rule():");
        System.out.println(rule.affich_text_rule(true));
        
        System.out.println("\nTest affich_nom_rules_ForDiversity():");
        System.out.println(rule.affich_nom_rules_ForDiversity());
    }
    
    private static void testComparisonMethods(Classification_Rule rule1, 
                                           Classification_Rule rule2,
                                           Classification_Rule rule3) {
        System.out.println("\nComparaison rule1 et rule2 (attend false):");
        System.out.println("isEqual: " + rule1.isEqual(rule2));
        System.out.println("isEqualwithoutCL: " + rule1.isEqualwithoutCL(rule2));
        
        System.out.println("\nComparaison rule1 et rule3 (attend true pour isEqualwithoutCL):");
        System.out.println("isEqual: " + rule1.isEqual(rule3));
        System.out.println("isEqualwithoutCL: " + rule1.isEqualwithoutCL(rule3));
    }
    
    private static void testWithRealDataset(Instances dataset) throws Exception {
        // Création d'une règle pour outlook=sunny
        Map<String, String> attrs = new HashMap<>();
        attrs.put("outlook", "sunny");
        
        ArrayList<Integer> instances = new ArrayList<>();
        for (int i = 0; i < dataset.numInstances(); i++) {
            if (dataset.instance(i).stringValue(0).equals("sunny")) {
                instances.add(i);
            }
        }
        
        Classification_Rule rule = Classification_Rule.createRule(dataset, instances, attrs);
        
        System.out.println("\nRègle créée à partir du dataset:");
        System.out.println("affich_nom_rule: " + rule.affich_nom_rule(true));
        System.out.println("affich_csv_rule: " + rule.affich_csv_rule(false));
        System.out.println("Pondération: " + rule.getRule_Ponderation());
    }
}