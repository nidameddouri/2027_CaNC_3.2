package weka.classifiers.fca;

import weka.core.*;

import java.util.ArrayList;
import java.util.Arrays;


public class Classify_Instance 
{
	
	/**
	 * Un constructeur
	 */
	public Classify_Instance() throws Exception{

	} 
	
	/**
	 * Cette fonction permet de classifier une instance nominale 
	 * @param inst : L'instance a predire ca classe
	 * @param regle: Une regle de classification dont les attributs sont nominaux
	 * @return L'indice de la classe attribuee pour l'instance inst
	 */
	public double classify_Instance_nom(Instance inst, Classification_Rule regle, boolean debug) throws Exception {
	    // Préparation de l'affichage
	    StringBuilder debugMsg = new StringBuilder();
	    debugMsg.append("\n=== Classification Debug ===\n");
	    debugMsg.append("Testing instance: ").append(instanceToString(inst)).append("\n");
	    debugMsg.append("Against rule: ").append(regle.affich_nom_rule(true)).append("\n");
	    debugMsg.append("--- Condition Verification ---\n");

	    double ind_class = -1.0;
	    boolean allConditionsMet = true;

	    // Vérification détaillée de chaque condition
	    for (int i = 0; i < inst.numAttributes() - 1; i++) {
	        String ruleValue = regle.Rule_Attr.get(i);
	        String instanceValue = inst.isMissing(i) ? "?" : inst.stringValue(i);
	        boolean conditionMet = ruleValue.equals("-null-") || ruleValue.equals(instanceValue);

	        debugMsg.append(String.format("Attr%d: Rule='%s' vs Instance='%s' -> %s%n",
	            i,
	            ruleValue,
	            instanceValue,
	            conditionMet ? "MATCH" : "MISMATCH"
	        ));

	        if (!ruleValue.equals("-null-") && !ruleValue.equals(instanceValue)) {
	            allConditionsMet = false;
	        }
	    }

	    // Résultat final
	    if (allConditionsMet) {
	        ind_class = regle.Rule_indClassMaj;
	        debugMsg.append(String.format("\nRESULT: Rule applied → Class '%s' (Index: %d)%n",
	            inst.classAttribute().value((int)ind_class),
	            (int)ind_class
	        ));
	    } else {
	        debugMsg.append("\nRESULT: Rule not applied (conditions not met)\n");
	    }

	   if ( debug) {
	        System.out.println(debugMsg.toString());
	   }

	    return ind_class;
	}

	// Méthode utilitaire pour formater l'instance
	private String instanceToString(Instance inst) {
	    StringBuilder sb = new StringBuilder();
	    for (int i = 0; i < inst.numAttributes(); i++) {
	        if (i > 0) sb.append(", ");
	        sb.append(inst.attribute(i).name())
	          .append("=")
	          .append(inst.isMissing(i) ? "?" : inst.stringValue(i));
	    }
	    return sb.toString();
	}
	
	/**
	 * Cette fonction permet de classifier une instance nominale 
	 * par une technique de vote pondere
	 * @param inst : L'instance a classifier
	 * @param ensemble_regles: Ensemble de regles
	 * @param nb_class : Nombre de classes
	 * @return L'indice de la classe predite pour l'instance inst
	 */
	public double classify_Instance_nom_VotePond(Instance inst, ArrayList<Classification_Rule> ensemble_regles, int nb_class, boolean debug) throws Exception	
	{
		//Selection selon le vote pondere		
		double [] tab_ponderation = new double [nb_class];
		for(int i=0; i<nb_class; i++)
			tab_ponderation[i] = (double) 0.0;
		
		// Parcourir les regles de classification
		boolean etat_classement=false;
		int tmp_ind_class;
		for(int i=0; i<ensemble_regles.size(); i++)
		{		
			if ( TestClassifInstance_nom(inst, ensemble_regles.get(i)) ) //Si une regle classifie cette instance nominale
			{
				etat_classement=true;
				tmp_ind_class = (int) ensemble_regles.get(i).Rule_indClassMaj;
				tab_ponderation[tmp_ind_class] = tab_ponderation[tmp_ind_class] + ensemble_regles.get(i).Rule_Ponderation;
			}
		}
		
		double ind_class;
		
		if(etat_classement)
		{
			//Rechercher l'indice de la classe la plus ponderee
			double tmp_ponderation;
			tmp_ponderation = -1;
			tmp_ind_class = -1;
			for(int i=0 ; i<nb_class ; i++) {
				if(tab_ponderation[i] > tmp_ponderation)
				{
					tmp_ponderation = tab_ponderation[i];
					tmp_ind_class = i;
				}	
				if (debug) {
					System.out.println(String.format(
				            "Règle %d appliquée -> Classe : %d, Poids ajouté : %.3f",
				            i, tmp_ind_class, ensemble_regles.get(i).Rule_Ponderation
				        ));	
				}
						}
			ind_class = (double) tmp_ind_class;
		}
		else
			ind_class = (double) -1.0;
		
		
		
		return ind_class ;			
	}
	
	/**
	 * Cette fonction permet de classifier une instance nominale 
	 * par une technique de vote majoritaire
	 * @param inst : L'instance a classifier
	 * @param ensemble_regles: Ensemble de regles
	 * @param nb_class : Nombre de classes
	 * @return L'indice de la classe predite pour l'instance inst
	 */
	
	public double classify_Instance_nom_VoteMaj(Instance inst, ArrayList<Classification_Rule> ensemble_regles, int nb_class, boolean debug) throws Exception {
    // Validation des entrées
    if (ensemble_regles == null || nb_class <= 0) {
        throw new IllegalArgumentException("Paramètres invalides");
    }
    if (debug) {
    	System.out.println("Rules number: " + ensemble_regles.size());
        System.out.println("[Debug] Nombre de classes: " + nb_class);
    }
    

    int[] tab_vote = new int[nb_class]; // Initialisé à 0 par défaut

    boolean etat_classement = false;
    int ruleIndex = 0; // Index pour le suivi des règles
    
    // Parcours des règles
    for (Classification_Rule regle : ensemble_regles) {
        if (TestClassifInstance_nom(inst, regle)) {
            etat_classement = true;
            int classe = (int) regle.Rule_indClassMaj;
            tab_vote[classe]++;
            
            System.out.println(String.format(
                "Règle %d appliquée -> Classe: %d, Vote ajouté. Total votes: %d",
                ruleIndex, classe, tab_vote[classe]
            ));
        } else {
            System.out.println(String.format(
                "Règle %d non appliquée", 
                ruleIndex
            ));
        }
        ruleIndex++;
    }

    if (!etat_classement ) {
    	if (debug) {
            System.out.println("No applicable rule!");

    	}
        return -1.0;
    }
   

    int maxVotes = 0;
    int classeMajoritaire = 0;
    for (int i = 0; i < nb_class; i++) {
        if (tab_vote[i] > maxVotes) {
            maxVotes = tab_vote[i];
            classeMajoritaire = i;
        }
    }
    if(debug) {
    	
    	 System.out.println("Votes per classe : " + Arrays.toString(tab_vote));
    }

   

    return (double) classeMajoritaire;
}
	
	/**
	 * Cette fonction permet de classifier une instance nominale 
	 * par une technique de vote a la pluralite
	 * Condition spéciale : La classe doit avoir >50% des votes 
	 * @param inst : L'instance a classifier
	 * @param ensemble_regles: Ensemble de regles
	 * @param nb_class : Nombre de classes
	 * @return L'indice de la classe predite pour l'instance inst
	 */
	public double classify_Instance_nom_VotePlur(Instance inst, ArrayList<Classification_Rule> ensemble_regles, int nb_class, boolean debug) throws Exception 
	{
		//Selection selon le vote majore		
		int [] tab_vote = new int [nb_class];
		for(int i=0; i<nb_class; i++)
			tab_vote[i] = (int) 0;
		
		// Parcourir les regles de classification
		boolean etat_classement=false;
		int tmp_ind_class=-1;
		for(int i=0; i<ensemble_regles.size(); i++)
		{		
			if (TestClassifInstance_nom(inst , ensemble_regles.get(i))) //Si une regle classifie cette instance nominale
			{
				etat_classement=true;
				tmp_ind_class = (int) ensemble_regles.get(i).Rule_indClassMaj; 
				tab_vote[tmp_ind_class] ++; 
			}
		}
		
		double ind_class;
		double seuilMajorite= (ensemble_regles.size() / 2.0);

		if(etat_classement)
		{
			//Rechercher l'indice de la classe la plus votee
			double tmp_vote;
			tmp_vote = -1;
			tmp_ind_class = -1;

			for(int i=0 ; i<nb_class ; i++)
				if(tab_vote[i] > tmp_vote)
				{
					tmp_vote = tab_vote[i];
					tmp_ind_class = i;
				}
			if(tab_vote[tmp_ind_class] >seuilMajorite)  // Compare en double				
				ind_class = (double) tmp_ind_class; 
			else
				ind_class = (double) -1.0;
		}
		else
			ind_class = (double) -1.0;
		
		
		if(debug) {
			System.out.println("Votes per classe : " + Arrays.toString(tab_vote));
			System.out.println("Majority threshold (Default) : " + seuilMajorite);
		}

		return ind_class ;			
	}

	/**
	 * Cette fonction permet de classifier une instance nominale 
	 * par une technique de vote a la pluralite avec un seuil variable
	 * @param inst : L'instance à classifier
	 * @param ensemble_regles: Ensemble de regles
	 * @param nb_class : Nombre de classes
	 * @param PlurMin : Seuil minimum pour la plurality d'un vote acceptable
	 * @return L'indice de la classe predite
	 */
	public double classify_Instance_nom_VotePlurMin(Instance inst, ArrayList<Classification_Rule> ensemble_regles, int nb_class, double PlurMin, boolean debug) 
			throws Exception 
	{	
		//Selection selon le vote majore		
		int [] tab_vote = new int [nb_class];
		for(int i=0; i<nb_class; i++)
			tab_vote[i] = (int) 0;
		
		// Parcourir les regles de classification
		boolean etat_classement=false;
		int tmp_ind_class=-1;
		for(int i=0; i<ensemble_regles.size(); i++)
		{		
			if (TestClassifInstance_nom(inst,ensemble_regles.get(i))) //Si une regle classifie cette instance nominale
			{
				etat_classement=true;
				tmp_ind_class = (int) ensemble_regles.get(i).Rule_indClassMaj; 
				tab_vote[tmp_ind_class] ++; 
			}
		}
		
		double ind_class;
		double conv_PlurMin=0.0;
		if(etat_classement)
		{
			//Rechercher l'indice de la classe la plus votee
			double tmp_vote;
			tmp_vote = -1;
			tmp_ind_class = -1;
			for(int i=0 ; i<nb_class ; i++)
				if(tab_vote[i] > tmp_vote)
				{
					tmp_vote = tab_vote[i];
					tmp_ind_class = i;
				}
			 conv_PlurMin = (double) PlurMin*ensemble_regles.size() / 100;
			if( tab_vote[tmp_ind_class] >= conv_PlurMin ) 
				ind_class = (double) tmp_ind_class; 
			else
				ind_class = (double) -1.0;
		}
		else
			ind_class = (double) -1.0;
		
		if (debug) {
			System.out.println("Votes per classe : " + Arrays.toString(tab_vote));
			System.out.println("Majority threshold (Set by user): " + conv_PlurMin);
			
    
		}

				return ind_class ;			
	}

	/**
	 * Cette fonction permet de classifier une instance nominale 
	 * par une technique de vote majoritaire par des classifieurs 
	 * qui depassent un seuil donne de ponderation (vote notoire)
	 * @param inst : L'instance a classifier
	 * @param ensemble_regles: Ensemble de regles
	 * @param nb_class : Nombre de classes
	 * @param LowPondVoter : Le seuil minimum du poids des classifieurs participant dans le vote
	 * @return L'indice de la classe predite 
	 */
	public double classify_Instance_nom_VoteLowPondVoter(Instance inst, ArrayList<Classification_Rule> ensemble_regles, int nb_class, double LowPondVoter, boolean debug) 
			throws Exception 
	{					
		// Creation d'un tableau pour vote majoritaire
		int [] tab_vote = new int [nb_class];
		for(int i=0; i<nb_class; i++)
			tab_vote[i] = (int) 0;
		
		double conv_LowPondVoter = (double) LowPondVoter / 100; 
		
		// Parcourir les regles de classification
		boolean etat_classement=false;
		int tmp_ind_class=-1;
		for(int i=0; i<ensemble_regles.size(); i++)
		{	
			//Si une regle classifie cette instance nominale
			if (ensemble_regles.get(i).getRule_Ponderation()>= conv_LowPondVoter && TestClassifInstance_nom(inst,ensemble_regles.get(i))) 
			{
				etat_classement=true;
				tmp_ind_class = (int) ensemble_regles.get(i).Rule_indClassMaj; 
				tab_vote[tmp_ind_class] ++; 
			}
		}
		
		double ind_class;
		
		if(etat_classement)
		{
			//Rechercher l'indice de la classe la plus votee
			double tmp_vote;
			tmp_vote = -1;
			tmp_ind_class = -1;
			for(int i=0 ; i<nb_class ; i++)
				if(tab_vote[i] > tmp_vote)
				{
					tmp_vote = tab_vote[i];
					tmp_ind_class = i;
				}
			ind_class = (double) tmp_ind_class; 
		}
		else
			ind_class = (double) -1.0;
				

		if (debug) {
			
			System.out.println("Votes per classe : " + Arrays.toString(tab_vote));
			System.out.println("Weight threshold: " + conv_LowPondVoter);
		}
		return ind_class ;			
	}

	/**
	 * Cette fonction permet de classifier une instance nominale 
	 * par une technique de vote pondere a partir d'un nombre bien
	 * determine du meilleurs classifieurs
	 * @param inst : L'instance a classifier
	 * @param ensemble_regles: Ensemble de regles
	 * @param nb_class : Nombre de classes
	 * @param MaxBestVoterS : Nombre des classifieurs
	 * @return L'indice de la classe predite pour l'instance inst
	 */
	public double classify_Instance_nom_VoteNumBestVoterS(Instance inst, ArrayList<Classification_Rule> ensemble_regles, int nb_class, int MaxBestVoterS, boolean debug) throws Exception {	
					
		// Creer un tableau a deux dimensions pour contenir les indices des classifieurs et leurs poids
		double [][] ens_reg_sorted = new double [ensemble_regles.size()][2];
		// Initialiser ens_reg_sorted[][] par les indices des classifieurs et leurs poids
		for(int i=0; i<ensemble_regles.size(); i++)
		{
			ens_reg_sorted[i][0] = (double) i;
			ens_reg_sorted[i][1] = (double) ensemble_regles.get(i).getRule_Ponderation();
		}
				
		// Trier ens_reg_sorted[][] par ordre decroissant		
		for(int i=0; i<ensemble_regles.size(); i++)
			for(int j=i+1; j<ensemble_regles.size(); j++)
			{
				if (ens_reg_sorted[j][1] > ens_reg_sorted[i][1])
				{
					double tmp_ind, tmp_pond;
					tmp_ind = ens_reg_sorted[j][0];
					tmp_pond = ens_reg_sorted[j][1];

					ens_reg_sorted[j][0] = i;
					ens_reg_sorted[j][1] = ens_reg_sorted[i][1];

					ens_reg_sorted[i][0] = tmp_ind;
					ens_reg_sorted[i][1] = tmp_pond;	
				}
			}

		// Selection selon le vote majore		
		int [] tab_vote = new int [nb_class];
		for(int i=0; i<nb_class; i++)
			tab_vote[i] = (int) 0;
		
		// Parcourir les regles de classification
		boolean etat_classement=false;
		int tmp_ind_class=-1;
		for(int i=0 ; i<ensemble_regles.size() && i<=(MaxBestVoterS-1) ; i++)
		{		
			if (TestClassifInstance_nom(inst , ensemble_regles.get((int) ens_reg_sorted[i][0]))) //Si une regle classifie cette instance nominale
			{
				etat_classement=true;
				tmp_ind_class = (int) ensemble_regles.get((int) ens_reg_sorted[i][0]).Rule_indClassMaj; 
				tab_vote[tmp_ind_class] ++; 
			}
		}
		
		// Rechercher l'indice de la classe la plus votee		
		double ind_class;
		if(etat_classement)
		{
			double tmp_vote;
			tmp_vote = -1;
			tmp_ind_class = -1;
			for(int i=0 ; i<nb_class ; i++)
				if(tab_vote[i] > tmp_vote)
				{
					tmp_vote = tab_vote[i];
					tmp_ind_class = i;
				}	
			ind_class = (double) tmp_ind_class; 
		}
		else
			ind_class = (double) -1.0;
		
		if (debug) {
			
			System.out.println("Votes par classe : " + Arrays.toString(tab_vote));
			System.out.println("Max classifiers Number: " + MaxBestVoterS);
		}
		
		
		
		return ind_class ;				
	}
	
	/**
	 * Parcourir les attributs nominaux de 'inst' et les comparer a ceux de 'regle'.
	 * Le but est de voir si 'regle' est applicable pour predire la classe de 'inst'
	 * Ignorer le dernier attribut qui represente la classe (inst.numAttributes()-1)
	 * Verifer que le i ieme attribut de la regle est different de '-null-' 
 	 * aussi different de la i eme attribut de l'instance nominale a classifier
	 * @param regle: La regle de classification
	 * @param inst: l'istance a classifier
	 * @return True ou False
	 */
	public boolean TestClassifInstance_nom(Instance inst, Classification_Rule regle) throws Exception {
	    // Vérification de cohérence
	    if (regle.Rule_Attr.size() != inst.numAttributes() - 1) {
	        throw new Exception("Rule/Instance attribute count mismatch");
	    }

	    for (int i = 0; i < inst.numAttributes() - 1; i++) {
	        String ruleValue = regle.Rule_Attr.get(i);
	        
	        if (ruleValue.equals("-null-")) {
	            continue;
	        }

	        // Gestion des valeurs manquantes dans l'instance
	        if (inst.isMissing(i)) {
	            return false;
	        }

	        // Comparaison robuste
	        String instanceValue = inst.stringValue(i);
	        if (!ruleValue.equals(instanceValue)) {
	            return false;
	        }
	    }
	    return true;
	}
	}
