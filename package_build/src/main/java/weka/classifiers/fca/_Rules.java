package weka.classifiers.fca;
import weka.core.*;

import java.util.ArrayList;

/**
 * @author Meddouri Nida (nmeddouri@gmail.com) 
 */

public class _Rules  {
	
public ArrayList <String> tab_attr_regle; // La liste des VALEURS d'attributs
public double ClasseMajoritaireRegle; // indice de la classe majoritaires
public double PonderationRegle;	// La valuer de la ponderation  


// Un constructeur null d'une r�gle
	public _Rules () {
		
		this.tab_attr_regle= new ArrayList <String>();
		this.ClasseMajoritaireRegle = (double) -1.0;
		this.PonderationRegle = (double) 0.0;	
	}
	
	// Un constructeur a partir de la liste des attribut, classe majoritaire et pond�ration.	
	public _Rules(ArrayList<String> tab, double CM, int critere){
		if(critere==0) // Binaire
		{			
			this.tab_attr_regle = new ArrayList <String>();
			this.ClasseMajoritaireRegle = CM;
			
			for(int i=0; i< tab.size();i++)		
		    {
				 this.tab_attr_regle.add(tab.get(i));
				 if (tab.get(i).substring(0,1).equals("1")) {
				 }  
		    }
			   
		} 
		else // Nominal
		{
			this.tab_attr_regle = new ArrayList <String>();
			this.ClasseMajoritaireRegle = CM;
			
		    for(int i=0; i< tab.size();i++)		
		    {
				 this.tab_attr_regle.add(tab.get(i));
		    }

		}
	}
	
	// Un constructeur a partir de la liste des attribut, classe majoritaire et ponderation.	
	public _Rules(ArrayList<String> tab,int CM, double ponderation, int critere){
		if(critere==0) // Binaire
		{			
			this.tab_attr_regle = new ArrayList <String>();
			this.ClasseMajoritaireRegle = CM;
			this.PonderationRegle = ponderation;
			
			for(int i=0; i< tab.size();i++)		
		    {
				 this.tab_attr_regle.add(tab.get(i));
				 if (tab.get(i).substring(0,1).equals("1")) {
				 }  
		    }
			   
		} 
		else // Nominal
		{
			this.tab_attr_regle = new ArrayList <String>();
			this.ClasseMajoritaireRegle = CM;
			this.PonderationRegle = ponderation;
			
		    for(int i=0; i< tab.size();i++)		
		    {
				 this.tab_attr_regle.add(tab.get(i));
		    }

		}
	}
	
	// Copier la regle this dans une autre en retour
	public _Rules copieRules ()
	{
		_Rules ob = new _Rules ();
		ob.tab_attr_regle = this.gettab_attr_regle();
		ob.ClasseMajoritaireRegle = this.getClasseMajoritaireRegle();
		ob.PonderationRegle = this.getPonderationRegle();	
	    
	    return ob;		
	}
	
	//Comparer la permisse de notre regle a l'instance 'inst'
	public boolean TestInstance (Instance inst){
		
		double [] att_instance= inst.toDoubleArray();
		 
		for (int i=0; i< inst.numAttributes()-1;i++)
		{
			String  c1=this.tab_attr_regle.get(i);
			if(  (c1.equals("1")) &&  (att_instance[i]==0)   )
			   return false;
		}
	return true;
	}
	
	public String toString  (ArrayList <String> rule)
	{
		
		 String chaine = new String ("IF "); 
		 
	     ArrayList <Integer> tab_indic_att = new ArrayList <Integer>();
	     int compteur_att = 0;
	    
	     //Extraction des indices des attributs
	     for (int j=0; j< this.tab_attr_regle.size(); j++)
	        	if (this.tab_attr_regle.get(j).substring(0,1).equals("1"))
	        	{ 
	        		tab_indic_att.add(j);
	        		compteur_att++;
	        	}
     	 compteur_att--; // Nombre des attributs de la regle est (compteur_att+1)
 
       	// Pour afficher les attributs sous format textuel
	     for (int g=0; g<compteur_att;g++)
	    	 chaine =chaine +rule.get(tab_indic_att.get(g))+ " AND";   	    	    
	    
	     chaine = chaine +rule.get(tab_indic_att.get(compteur_att));  	// Pour afficher le dernier attribut sous format textuel     
	     chaine = chaine +" ";
	     chaine = chaine +" THEN  Class  "+ this.getClasseMajoritaireRegle();
	     chaine = chaine +" WITH Ponderation =  "+ Utils.roundDouble(this.getPonderationRegle(),2);
	       
		return chaine;
	}
		
	// Affichage binaire de la pr�misse de la r�gle. Si x est vrai on associe une classe majoritaire sinon sans classe
	public String affich_binaire_regles (boolean AfficheCM, Instances instances, ArrayList<ArrayList<String>> EtiquetteClass){
		
		String chaine = new String();
		int index;
		
		if (tab_attr_regle.size()!=0)
		{
		chaine = chaine+" IF ";
		index = Integer.parseInt(this.tab_attr_regle.get(0));
		
		if (index == 1)
		{
			chaine = chaine+ instances.attribute(0).name()+" ";
		}
		
		for (int j=1; j< this.tab_attr_regle.size(); j++)
		{
			index = Integer.parseInt(this.tab_attr_regle.get(j));
			if (index == 1)
			{
				if (chaine.compareTo(" IF ")==0)
					chaine = chaine+instances.attribute(j).name();
				else
					chaine = chaine+" AND "+instances.attribute(j).name();
			}
				
		}
		}// fin if (tab_attr_regle)    	     
	     chaine = chaine+"  THEN  ";
	    
	     if (AfficheCM)
	     {
	    	 boolean existe=false;
		    	int indexEtti=-1;
		    	while ((indexEtti<EtiquetteClass.size()-1)&&(!existe))
		    	{
		    		indexEtti++;
		    		if(Integer.parseInt(EtiquetteClass.get(indexEtti).get(0))==this.getClasseMajoritaireRegle())
		    			existe=true;
		      	}
		    	chaine = chaine+EtiquetteClass.get(indexEtti).get(1);
	     }    
	        chaine = chaine + " Ponderation: " + Utils.roundDouble(this.getPonderationRegle(),2); 
	        
		return chaine;
	}
	
	// Affichage nominal de la pr�misse de la regle. Si x est vrai on associe une classe majoritaire sinon sans classe
	public String affich_nom_regles (boolean AfficheCM){
		
		String chaine = new String ("");
		for (int j=0; j< this.tab_attr_regle.size(); j++)
     	   	chaine = chaine+this.tab_attr_regle.get(j)+","	;        	
     	     	     
	     chaine=chaine+"  ";
	     if (AfficheCM)
		        chaine = chaine +" Indice classe: "+this.getClasseMajoritaireRegle()+"  ";
	          
	        chaine = chaine + " Ponderation: " + Utils.roundDouble(this.getPonderationRegle(),2); 
	        
		return chaine;
	}
	
	// Affichage nominal de la premisse de la regle. Si x est vrai on associe une classe majoritaire sinon sans classe
	public String affich_nom_regles_ForDiversity (){
		
		String chaine=new String ("");
		for (int j=0; j< this.tab_attr_regle.size()-1; j++)
     	   	chaine=chaine+this.tab_attr_regle.get(j)+",";        	
		chaine=chaine+this.tab_attr_regle.get(this.tab_attr_regle.size()-1);
     	     	     
		return chaine;
	}
	
	// Comparer cette regle a une autre y compris la partie conclusion
	 public  boolean isEqual (_Rules rl)
     {
 	 
 	 for (int i=0 ; i< rl.gettab_attr_regle().size(); i++)
	         if ( ! this.tab_attr_regle.get(i).equals(rl.gettab_attr_regle().get(i)))
	         return false;	 
 	  
 	 if ( this.ClasseMajoritaireRegle != rl.getClasseMajoritaireRegle ())
 		 return false;
 	 
 	return  true ;	 
 	
      }

	// Comparer cette regle a une autre sans la partie conclusion
	 public  boolean isEqualwithoutCL (_Rules rl)
     {
 	 
 	 for (int i=0 ; i< rl.gettab_attr_regle().size(); i++)
	         if ( ! this.tab_attr_regle.get(i).equals(rl.gettab_attr_regle().get(i)))
	         return false;
 	 
 	return  true ;	 
 	
      }
	
	
	public void setPonderationRegle(double PonderationRegle) 
	{
		this.PonderationRegle = PonderationRegle;
	}

	public void setClasseMajoritaireRegle( double ClasseMajoritaireRegle)
	{
		this.ClasseMajoritaireRegle = ClasseMajoritaireRegle;
	}
	
	public void settab_attr_regle(ArrayList<String> tab) 
	{
		this.tab_attr_regle = new ArrayList <String>();
		for(int i=0; i< tab.size();i++)
			     this.tab_attr_regle.add(tab.get(i));
	}


	public double getPonderationRegle() 
	{
		return PonderationRegle;
	}

	public ArrayList<String> gettab_attr_regle() 
	{
		return tab_attr_regle;
	}
	
	public  double getClasseMajoritaireRegle() 
	{
		return this.ClasseMajoritaireRegle;
	}
	
	}
