package weka.classifiers.fca;
import weka.core.*;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;

public class Classification_Rule {
    // Changement ici: utilisation d'une Map pour stocker les attributs
    private Map<String, String> attributeValues; // Clé=nom attribut, Valeur=valeur
    private double rule_indClassMaj;
    private String rule_ClassMaj;
    private double rule_Ponderation;
    
    public Classification_Rule() {
        this.attributeValues = new HashMap<>();
        this.rule_indClassMaj = -1.0;
        this.rule_ClassMaj = "";
        this.rule_Ponderation = -1.0;
    }
    
    // Constructeur avec initialisation des valeurs
    public Classification_Rule(Map<String, String> attributes, 
                             double ind_CM, 
                             String CM) {
        this.attributeValues = new HashMap<>(attributes);
        this.rule_indClassMaj = ind_CM;
        this.rule_ClassMaj = CM;
        this.rule_Ponderation = -1.0;
    }
    
    // Constructeur avec initialisation des valeurs avec pods
    public Classification_Rule(Map<String, String> attributes, 
                             double ind_CM, 
                             String CM,
                             double poids) {
        this.attributeValues = new HashMap<>(attributes);
        this.rule_indClassMaj = ind_CM;
        this.rule_ClassMaj = CM;
        this.rule_Ponderation = poids;
    }
    
    // Méthode pour créer une règle avec pondération
    public static Classification_Rule createRule(Instances inst, 
                                               ArrayList<Integer> instanceIndices,
                                               Map<String, String> attributes) throws Exception {
        // Trouver la classe majoritaire
        int[] classCounts = new int[inst.numClasses()];
        for (int idx : instanceIndices) {
            classCounts[(int)inst.instance(idx).classValue()]++;
        }
        
        int majorityClass = 0;
        for (int c = 1; c < classCounts.length; c++) {
            if (classCounts[c] > classCounts[majorityClass]) {
                majorityClass = c;
            }
        }
        
        double weight = (double)instanceIndices.size() / inst.numInstances();
        
        Classification_Rule rule = new Classification_Rule(
            attributes,
            majorityClass,
            inst.classAttribute().value(majorityClass)
        );
        rule.setRule_Ponderation(weight);
        
        return rule;
    }
    
    
    /**
	 * Version sécurisée qui vérifie toutes les règles pour trouver le max
	 */
	public static int getMaxNumClass(ArrayList<Classification_Rule> rules) {
	    int maxClass = -1;
	    for (Classification_Rule rule : rules) {
	        if (rule.getRule_indClassMaj() > maxClass) {
	            maxClass = (int)rule.getRule_indClassMaj();
	        }
	    }
	    return maxClass + 1; // conversion 0-based → count
	}

    // Getter pour les prémisses (format "nom=valeur")
    public ArrayList<String> getPremisses() {
        ArrayList<String> premisses = new ArrayList<>();
        for (Map.Entry<String, String> entry : attributeValues.entrySet()) {
            premisses.add(entry.getKey() + "=" + entry.getValue());
        }
        return premisses;
    }
    
    // Affichage amélioré de la règle
    public String affich_nom_rule(boolean afficheCM) {
        StringBuilder sb = new StringBuilder("IF ");
        
        boolean first = true;
        for (Map.Entry<String, String> entry : attributeValues.entrySet()) {
            if (!first) sb.append(" AND ");
            sb.append(entry.getKey()).append("=").append(entry.getValue());
            first = false;
        }
        
        if (afficheCM) {
            sb.append(" THEN ").append(rule_ClassMaj)
              .append(" [").append(rule_indClassMaj).append("]");
        }
        
        sb.append(" (Weight: ").append(Utils.roundDouble(rule_Ponderation, 3)).append(")");
        
        return sb.toString();
    }
  
	
	

	/**
	 * Returns the number of valid premises in the rule
	 */
	public int getPremissesCount() {
	  
	    return this.attributeValues.size();
	}
	
	
	// Affichage CSV des prémisses de la règle
	public String affich_csv_rule(boolean AfficheCM) {
	    StringBuilder chaine = new StringBuilder("\t");
	    
	    // Parcours des attributs dans la Map
	    boolean first = true;
	    for (Map.Entry<String, String> entry : attributeValues.entrySet()) {
	        if (!first) chaine.append(",");
	        chaine.append(entry.getKey()).append("=").append(entry.getValue());
	        first = false;
	    }

	    if (AfficheCM) {
	        chaine.append(", ").append(this.getRule_ClassMaj());
	    }
	    chaine.append(",").append(Utils.roundDouble(this.getRule_Ponderation(), 3));
	    
	    return chaine.toString();
	}

	// Affichage texte détaillé de la règle
	public String affich_text_rule(boolean AfficheCM) {
	    StringBuilder chaine = new StringBuilder(" ");
	    
	    boolean first = true;
	    for (Map.Entry<String, String> entry : attributeValues.entrySet()) {
	        if (!first) chaine.append(" , ");
	        chaine.append(entry.getKey()).append("=").append(entry.getValue());
	        first = false;
	    }

	    if (AfficheCM) {
	        chaine.append(" |==> Class=").append(this.getRule_ClassMaj());
	    }
	    chaine.append(" | ").append(Utils.roundDouble(this.getRule_Ponderation(), 3));
	    
	    return chaine.toString();
	}

	// Affichage simplifié pour la diversité
	public String affich_nom_rules_ForDiversity() {
	    StringBuilder chaine = new StringBuilder();
	    
	    boolean first = true;
	    for (Map.Entry<String, String> entry : attributeValues.entrySet()) {
	        if (!first) chaine.append(",");
	        chaine.append(entry.getValue());
	        first = false;
	    }
	    
	    return chaine.toString();
	}
	// Comparaison complète avec une autre règle
	public boolean isEqual(Classification_Rule rl) {
	    // Comparaison des attributs
	    if (!this.attributeValues.equals(rl.getAttributeValues())) {
	        return false;
	    }
	    
	    // Comparaison de la classe
	    return this.rule_indClassMaj == rl.getRule_indClassMaj();
	}

	// Comparaison sans la conclusion
	public boolean isEqualwithoutCL(Classification_Rule rl) {
	    return this.attributeValues.equals(rl.getAttributeValues());
	}
	public void setRule_Ponderation(double Rule_Ponderatio) {
	    this.rule_Ponderation = Rule_Ponderatio;
	}

	public void setRule_indClassMaj(double Rule_indClassMa) {
	    this.rule_indClassMaj = Rule_indClassMa;
	}

	public void setRule_ClassMaj(String Rule_ClassMa) {
	    this.rule_ClassMaj = Rule_ClassMa;
	}

	public void setAttributeValues(Map<String, String> values) {
	    this.attributeValues = new HashMap<>(values);
	}

	public Map<String, String> getAttributeValues() {
	    return new HashMap<>(this.attributeValues);
	}

	public double getRule_Ponderation() {
	    return rule_Ponderation;
	}

	public double getRule_indClassMaj() {
	    return rule_indClassMaj;
	}

	public String getRule_ClassMaj() {
	    return rule_ClassMaj;
	}
	// Ajoute un attribut à la règle
	public void addAttribute(String name, String value) {
	    if (name != null && value != null && !value.equals("-null-")) {
	        this.attributeValues.put(name, value);
	    }
	}
	
	}