
#This part we follow the previous sota method "Leveraging type descriptions for zero-shot named entity recognition and classification  acl 2021"
#We use the same description and description creation method, the specific construction method can be found in 3.2 and appendix 6.1 and 6.2 of the "Leveraging type descriptions for zero-shot named entity recognition and classification."

_MedMention_tag_instantiation={
    "Biologic Function":'A biological function is the reason some object or process occurred in a system that evolved through a process of selection or natural selection. such as regulation,regulate.',
    'Chemical':'Chemical means involving or resulting from a reaction between two or more compounds, or relating to the substances that something consists of. ',
    "Health Care Activity":"A Health facility is, in general, any location where health care is provided. Health facilities range from small clinics and doctor's offices to urgent care centers and large hospitals with elaborate emergency rooms and trauma centers.",
    "Anatomical Structure":"An anatomical structure is any biological entity that occupies space and is distinguished from its surroundings. cadavers,whole body",
    "Finding":"Someone's findings are the information they get or the conclusions they come to as the result of an investigation or some research results.",
    "Spatial Concept":"spatial concepts are mainly expressed by means of metaphors, namely spatial-cognitive metaphors,such as pores",
    "Intellectual Product":"The intellectual products by human beings can be classified into that in material form and that reflected by his activities. such as psychological theory",
    "Research Activity":"Research Activity means ascientific investigation or inquiry that results in the generation of knowledge.",
    "Eukaryote":"an organism with cells characteristic of all life forms except primitive microorganisms such as bacteria; i.e. an organism with 'good' or membrane-bound nuclei in its cells",
    "Population Group":"Population group means one or more demographic groups to which the person belongs, and white, South Asian, female, or national groups are all different types of population groups.",
    "Medical Device":"A medical device is an instrument, apparatus, implant, in vitro reagent, or similar or related article that is used to diagnose, prevent, or treat disease or other conditions, and does not achieve its purposes through chemical action within or on the body.",
    "Organization":" An organization is an official group of people,  a political party and a public institutions.",
    "Injury or Poisoning":"Poisoning is the harmful effect that occurs when a toxic substance is swallowed, is inhaled, or comes in contact with  the mouth or nose injury, while injury is harmful effect,trauma.",
    "Clinical Attribute":"Clinical attributes were some indicators of patients such as heart rate, blood pressure,markers, biomarkers etc. Clinical refers to direct contact with the patient for actual observation of the patient. ",
    "Virus":"A virus is a submicroscopic infectious agent that replicates only inside the living cells of an organism. Viruses infect all life forms, from animals and plants to microorganisms, including bacteria, archaea, HIV and recombinant baculovirus.",
    "Biomedical Occupation or Discipline":"Biomedicine (also referred to as Western medicine, mainstream medicine or conventional medicine)is a branch of medical science that applies biological and physiological principles to clinical practice. ",
    "Bacterium":"Bacteria are ubiquitous, mostly free-living organisms often consisting of one biological cell. They constitute a large domain of prokaryotic microorganisms.",
    "Professional or Occupational Group":"A professional or occupational group refers to a special group of people or groups of a professional nature.",
    "Food":"Food is any substance consumed by an organism for nutritional support. Food refers carbohydrates, pulp, or minerals, which are usually of plant, animal, or fungal origin and contain essential nutrients.",
    "Body Substance":"Body substances are physiological anatomical entities. Urine and blood are the main body substances, which are liquid and semi-solid.",
    "Body System":"A biological system is a complex network which connects several biologically relevant entities. Body system are determined based different structures depending on what the system is.",
}


_OntoNotes_tag_instantiation = {
    "CARDINAL": "CARDINAL is only marked if it is a numerical and not a year nor ORDINAL, MONEY, PERCENTAGE, nor QUANTITY.",
    "DATE": "Replace entities of type date with date. A date is a specific time that can be named, for example, a particular day or a particular year. ",
    "EVENT": "Named hurricanes, battles, wars, sports events, attacks.  Metonymic mentions (marked with a ∼) of the date or location of an event, or of the organization(s) involved, are included. ",
    "FAC": "Names of man-made structures: infrastructure (streets, bridges), buildings, monuments, etc. belong to this type. Buildings are referred to using the name of the company or organization.",
    "GPE": "Replace entities of type geographical social political entity with geographical social political entity. China National, provincial and municipal administrative regions. ",
    "LANGUAGE": " English Frequent languages (English, German,...) and if preceded by [in, into, speak, write, talk, listen, ...]. ",
    "LAW": "Any document that has been made into a law, including named treaties and sections and chapters of named legal documents.",
    "LOC": "These include mountain ranges, coasts, borders, planets, geocoordinates, bodies of water. Also included in this category are named regions such as the Middle East, areas, neighborhoods, continents and regions of continents. ",
    "MONEY": "Any token dollars, euro, yuan, pound, ... or its symbolic representation and preceding numbers, incl. [hundred(s), thousand(s), million(s), billion(s)].",
    "NORP": "Nationality religion political refers to ethnic and relplaces [Palestiigious politics, holy nian,Japanese]The entity type of Nationality religion political is",
    "ORDINAL": "Any word that is either 'first', 'second', or 'third', or compound of 'th' and a number.",
     "ORG": "Replace entities of type organization with organization. An organization is an official group of people, for example, a political party, a business, a charity, or a club.",
    "PERCENT": "Any token that is % and its preceding number, as well as the preceding adverb such as 'about', 'around', or 'approximately'. ",
    "PERSON": "Replace entities of type person with person. A person is a man, woman, or child, such as Tom. ",
    "PRODUCT": "A product is something that is produced and sold in large quantities, often as a result of a manufacturing process. ",
    "QUANTITY": "One of ca. 20 SI units (incl. its abbreviation) and preceding number and relevant adverb/preposition.",
    "TIME": "'a.m', 'p.m.', 'morning', 'evening', 'night', 'minute(s)', 'hour(s)' etc. and any preceding or consecutive numerical and relevant adverb/preposition. ",
    "WORK_OF_ART": "Titles of books, songs, television programs and other creations. Also includes awards. These are usually surrounded by quotation marks in the article (though the quotations are not included in the annotation). A work of art is a painting or piece of sculpture of high quality, and something that is attractive and skilfully made."
}

def MedMention_tag_instantiation():
    return _MedMention_tag_instantiation

def OntoNotes_tag_instantiation():
    return _OntoNotes_tag_instantiation