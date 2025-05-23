package weka.classifiers.fca;

import java.util.Set;

public class FormalConcept {
  	private Set<String> m_Objects;
    private Set<String> m_Attributes;

     public FormalConcept(Set<String> objects, Set<String> attributes) {
         this.m_Objects = objects;
         this.m_Attributes = attributes;
     }

     @Override
     public String toString() {
         return
                "Objects: " + m_Objects +
                "  Attributes: " + m_Attributes;
     }
     
     public Set<String> getObjects()
     {
    	 return m_Objects;
     }
     
     
     public Set<String> getAttributes()
     {
    	 return m_Attributes;
     }
     
     // Check if this concept is a parent of another concept
     public boolean isParentOf(FormalConcept other) {
         return this.m_Objects.containsAll(other.getObjects()) && 
                other.getAttributes().containsAll(this.m_Attributes);
     }
     
     // Get the level in the lattice (number of attributes)
     public int getLevel() {
         return m_Attributes.size();
     }
     
     @Override
     public boolean equals(Object obj) {
         if (!(obj instanceof FormalConcept)) return false;
         FormalConcept other = (FormalConcept) obj;
         return m_Objects.equals(other.m_Objects) && m_Attributes.equals(other.m_Attributes);
     }
     
     @Override
     public int hashCode() {
         return m_Objects.hashCode() + m_Attributes.hashCode();
     }
     

}


