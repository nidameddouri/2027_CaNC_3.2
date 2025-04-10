package weka.classifiers.fca;

import weka.core.*;

import java.util.ArrayList;


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
	public double classify_Instance_nom(Instance inst, Classification_Rule regle) throws Exception	{	
		
		double ind_class = (double) -1.0;
		if (TestClassifInstance_nom(inst,regle)) //Si la regle classifie cette instance nominale
				ind_class = regle.Rule_indClassMaj;		
		return ind_class ;		
	}
	
	/**
	 * Cette fonction permet de classifier une instance nominale 
	 * par une technique de vote pondere
	 * @param inst : L'instance a classifier
	 * @param ensemble_regles: Ensemble de regles
	 * @param nb_class : Nombre de classes
	 * @return L'indice de la classe predite pour l'instance inst
	 */
	public double classify_Instance_nom_VotePond(Instance inst, ArrayList<Classification_Rule> ensemble_regles, int nb_class) throws Exception	
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
			for(int i=0 ; i<nb_class ; i++)
				if(tab_ponderation[i] > tmp_ponderation)
				{
					tmp_ponderation = tab_ponderation[i];
					tmp_ind_class = i;
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
	public double classify_Instance_nom_VoteMaj(Instance inst, ArrayList<Classification_Rule> ensemble_regles, int nb_class) throws Exception 
	{	
		//Selection selon le vote majore	
		int [] tab_vote = new int [nb_class];
		for(int i=0; i<nb_class; i++)
			tab_vote[i] = (int) 0;
		
		// Parcourir les regles de classification
		boolean etat_classement=false;
		int tmp_ind_class;
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
		
		return ind_class ;			
	}

	/**
	 * Cette fonction permet de classifier une instance nominale 
	 * par une technique de vote a la pluralite
	 * @param inst : L'instance a classifier
	 * @param ensemble_regles: Ensemble de regles
	 * @param nb_class : Nombre de classes
	 * @return L'indice de la classe predite pour l'instance inst
	 */
	public double classify_Instance_nom_VotePlur(Instance inst, ArrayList<Classification_Rule> ensemble_regles, int nb_class) throws Exception 
	{
		//Selection selon le vote majore		
		int [] tab_vote = new int [nb_class];
		for(int i=0; i<nb_class; i++)
			tab_vote[i] = (int) 0;
		
		// Parcourir les regles de classification
		boolean etat_classement=false;
		int tmp_ind_class;
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
			
			if( tab_vote[tmp_ind_class] > (ensemble_regles.size()/2) )
				ind_class = (double) tmp_ind_class; 
			else
				ind_class = (double) -1.0;
		}
		else
			ind_class = (double) -1.0;
		
		return ind_class ;			
	}

	/**
	 * Cette fonction permet de classifier une instance nominale 
	 * par une technique de vote a la pluralite avec un seuil variable
	 * @param inst : L'instance à classifier
	 * @param ensemble_regles: Ensemble de regles
	 * @param nb_class : Nombre de classes
	 * @param PlurMin : Seuil minimum pour la pluralite d'un vote acceptable
	 * @return L'indice de la classe predite
	 */
	public double classify_Instance_nom_VotePlurMin(Instance inst, ArrayList<Classification_Rule> ensemble_regles, int nb_class, double PlurMin) 
			throws Exception 
	{	
		//Selection selon le vote majore		
		int [] tab_vote = new int [nb_class];
		for(int i=0; i<nb_class; i++)
			tab_vote[i] = (int) 0;
		
		// Parcourir les regles de classification
		boolean etat_classement=false;
		int tmp_ind_class;
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
			double conv_PlurMin = (double) PlurMin*ensemble_regles.size() / 100;
			if( tab_vote[tmp_ind_class] >= conv_PlurMin ) 
				ind_class = (double) tmp_ind_class; 
			else
				ind_class = (double) -1.0;
		}
		else
			ind_class = (double) -1.0;
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
	public double classify_Instance_nom_VoteLowPondVoter(Instance inst, ArrayList<Classification_Rule> ensemble_regles, int nb_class, double LowPondVoter) 
			throws Exception 
	{					
		// Creation d'un tableau pour vote majoritaire
		int [] tab_vote = new int [nb_class];
		for(int i=0; i<nb_class; i++)
			tab_vote[i] = (int) 0;
		
		double conv_LowPondVoter = (double) LowPondVoter / 100; 
		
		// Parcourir les regles de classification
		boolean etat_classement=false;
		int tmp_ind_class;
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
	public double classify_Instance_nom_VoteNumBestVoterS(Instance inst, ArrayList<Classification_Rule> ensemble_regles, int nb_class, int MaxBestVoterS) throws Exception {	
					
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
		int tmp_ind_class;
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
	public boolean TestClassifInstance_nom(Instance inst , Classification_Rule regle) throws Exception 
	{
		String nl = "-null-";
		for (int i=0 ; i<inst.numAttributes()-1 ; i++)
			{
			if(regle.Rule_Attr.get(i).compareTo(nl)!=0 
			&& regle.Rule_Attr.get(i).compareTo(inst.stringValue(i))!=0)
			{
				return false;
			}
		}
		return true;
	}

}
