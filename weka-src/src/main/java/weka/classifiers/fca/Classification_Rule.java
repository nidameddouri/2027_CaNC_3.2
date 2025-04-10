package weka.classifiers.fca;
import weka.core.*;

import java.util.ArrayList;

public class Classification_Rule  {
	
	public ArrayList <String> Rule_Attr; // La liste des VALEURS d'attributs
	public double Rule_indClassMaj; // indice de la classe majoritaire.
	public String Rule_ClassMaj; // Classe majoritaire.
	public double Rule_Ponderation;	// La valeur de la ponderation.
	private ArrayList<String> attributeNames; // Noms des attributs pour l'affichage

// Un constructeur null d'une regle
	public Classification_Rule () {
		this.Rule_Attr 			= new ArrayList <String>();
		this.Rule_indClassMaj 	= (double) -1.0;
		this.Rule_ClassMaj 		= "";
		this.Rule_Ponderation 	= (double) -1.0;
		this.attributeNames     = new ArrayList<String>();
	}
	
	// Un constructeur a partir de la liste des attributs, indice et classe majoritaire.	
	public Classification_Rule(ArrayList<String> attributes, double ind_CM, String CM){
		this.Rule_Attr 			= new ArrayList <String>();
		this.Rule_indClassMaj 	= (double) ind_CM;
		this.Rule_ClassMaj 		= CM;
		this.attributeNames     = new ArrayList<String>();
		
		for(int i=0; i< attributes.size();i++)	
			 this.Rule_Attr.add(attributes.get(i));
	}
	
	// Un constructeur a partir de la liste des attribut, indice et classe majoritaire et ponderation.	
	public Classification_Rule(ArrayList<String> attributes, double ind_CM, String CM, double ponderation){
		this.Rule_Attr 			= new ArrayList <String>();
		this.Rule_indClassMaj	= (double) ind_CM;
		this.Rule_ClassMaj 		= CM;
		this.Rule_Ponderation 	= ponderation;
		this.attributeNames     = new ArrayList<String>();
		
		for(int i=0; i< attributes.size();i++)		
			this.Rule_Attr.add(attributes.get(i));		
	}
	
	// Copier la regle this dans une autre en retour
	public Classification_Rule copieRules ()	{
		Classification_Rule ob 	= new Classification_Rule ();
		ob.Rule_Attr 			= this.getRule_Attr();
		ob.Rule_indClassMaj 	= this.getRule_indClassMaj();
		ob.Rule_ClassMaj 		= this.getRule_ClassMaj();
		ob.Rule_Ponderation 	= this.getRule_Ponderation();
		ob.attributeNames       = this.getAttributeNames();
	    return ob;		
	}

	// Affichage nominal de la regle.
	public String affich_nom_rule(boolean AfficheCM) {
		StringBuilder chaine = new StringBuilder("IF ");
		boolean first = true;
	
		for (int i = 0; i < this.Rule_Attr.size(); i++) {
			String attr = this.Rule_Attr.get(i);
			if (attr != null && !attr.equals("-null-")) {
				if (!first) {
					chaine.append(" ^ ");
				}
				
				// Ajout du nom de l'attribut avant sa valeur
				String attrName = (i < this.attributeNames.size()) ? this.attributeNames.get(i) : "attr" + i;
				chaine.append(attrName).append("=").append(attr);
				first = false;
			}
		}
	
		if (AfficheCM) {
			chaine.append(" THEN ").append(this.getRule_ClassMaj())
				  .append("\t Indice classe: ").append(this.getRule_indClassMaj());
		} else if (!first) {
			// If we displayed some attributes but not AfficheCM
			chaine.append(" THEN ").append(this.getRule_ClassMaj());
		}
	
		chaine.append("\t Ponderation: ")
			  .append(Utils.roundDouble(this.getRule_Ponderation(), 3));
	
		return chaine.toString();
	}
	

	// Affichage Comma-separated values de la premisse de la regle. Si AfficheCM est vraie, on associe une classe majoritaire sinon sans classe
	public String affich_csv_rule (boolean AfficheCM){
			
		String chaine = new String ("\t");
		for (int j=0; j< this.Rule_Attr.size(); j++)
			if(j!=this.Rule_Attr.size()-1)
     	   		chaine = chaine+this.Rule_Attr.get(j)+",";
			else
				chaine = chaine+this.Rule_Attr.get(j);

	     if (AfficheCM)
		        chaine = chaine + ", " + this.getRule_ClassMaj();
	     chaine = chaine + "," + Utils.roundDouble(this.getRule_Ponderation(),3); 
	        
		return chaine;
	}
	

	// Affichage Comma-separated values de la premisse de la regle. Si AfficheCM est vraie, on associe une classe majoritaire sinon sans classe
	public String affich_text_rule (ArrayList <String> attributesList, boolean AfficheCM){
			
		String chaine = new String (" ");
		for (int j=0; j< this.Rule_Attr.size(); j++)
			if(Rule_Attr.get(j).equals("-null-")==false && Rule_Attr.get(j).equals("'All'")==false)
				if(j!=this.Rule_Attr.size()-1)
     	   			chaine = chaine+attributesList.get(j)+"="+this.Rule_Attr.get(j)+" , ";
				else
					chaine = chaine+attributesList.get(j)+"="+this.Rule_Attr.get(j);

	     if (AfficheCM)
		        chaine = chaine + " |==> " +attributesList.get(attributesList.size()-1)+"="+ this.getRule_ClassMaj();
	     chaine = chaine + " | " + Utils.roundDouble(this.getRule_Ponderation(),3); 
	        
		return chaine;
	}
	
	// Affichage nominal de la pr�misse de la r�gle.
	public String affich_nom_rules_ForDiversity (){
		String chaine=new String ("");
		for (int j=0; j< this.Rule_Attr.size()-1; j++)
     	   	chaine = chaine+this.Rule_Attr.get(j)+",";        	
		chaine = chaine+this.Rule_Attr.get(this.Rule_Attr.size()-1);
		return chaine;
	}
	
	// Comparer cette regle a une autre y compris la partie conclusion
	 public boolean isEqual (Classification_Rule rl){
		 for (int i=0 ; i< rl.getRule_Attr().size(); i++)
			 if ( ! this.Rule_Attr.get(i).equals(rl.getRule_Attr().get(i)))
				 return false;
		 
		 if ( this.Rule_indClassMaj != rl.getRule_indClassMaj())
			 return false;
		 
		 return true ;
		 }

	// Comparer cette regle a une autre sans la partie conclusion
	 public boolean isEqualwithoutCL (Classification_Rule rl){
		 for (int i=0 ; i< rl.getRule_Attr().size(); i++)
			 if ( ! this.Rule_Attr.get(i).equals(rl.getRule_Attr().get(i)))
				 return false;
		 return  true ;	 
 	
      }
	
	
	public void setRule_Ponderation(double Rule_Ponderatio) {
		this.Rule_Ponderation = Rule_Ponderatio;
		}

	public void setRule_indClassMaj( double Rule_indClassMa){		
		this.Rule_indClassMaj = Rule_indClassMa;
		}

	public void setRule_ClassMaj( String Rule_ClassMa){		
		this.Rule_ClassMaj = Rule_ClassMa;
		}
	
	public void setRule_Attr(ArrayList<String> tab){
		this.Rule_Attr =new ArrayList <String>();
		for(int i=0; i< tab.size();i++)
			     this.Rule_Attr.add(tab.get(i));
		}

	// Définir les noms des attributs
	public void setAttributeNames(ArrayList<String> names) {
		this.attributeNames = new ArrayList<String>();
		for(int i=0; i < names.size(); i++) {
			this.attributeNames.add(names.get(i));
		}
	}
	
	// Récupérer les noms des attributs
	public ArrayList<String> getAttributeNames() {
		return attributeNames;
	}

	public double getRule_Ponderation() 	{
		return Rule_Ponderation;
		}

	public ArrayList<String> getRule_Attr() {
		return Rule_Attr;
		}
	
	public  double getRule_indClassMaj(){
		return this.Rule_indClassMaj;
		}
	
	public  String getRule_ClassMaj()	{
		return this.Rule_ClassMaj;
		}
	
	}