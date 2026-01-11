export interface Patient {
    id: string
    name: string
    age: number
    gender: "Male" | "Female" | "Other"
    dateOfBirth: string
    phone: string
    email: string
    address: string
    bloodType: string
    insuranceProvider: string
    insuranceNumber: string
    emergencyContact: {
        name: string
        relationship: string
        phone: string
    }
    symptoms: string[]
    reasonForVisit: string
    vitalSigns: {
        bloodPressure: string
        heartRate: number
        temperature: number
        oxygenSaturation: number
        weight: number
        height: number
    }
    medicalHistory: string[]
    allergies: string[]
    currentMedications: string[]
    lastVisit: string
    status: "Waiting" | "In Progress" | "Completed"
    priority: "Low" | "Medium" | "High" | "Critical"
    photo?: string
}

export async function loadPatientsFromFilesystem(): Promise<Patient[]> {
    try {
        const response = await fetch('http://localhost:3001/list-patients');
        if (!response.ok) {
            console.error('Failed to load patients from filesystem');
            return patients; // Fallback to mock data
        }
        
        const data = await response.json();
        console.log('📦 Raw API response:', data);
        
        // Transform filesystem data to Patient interface
        const transformed = data.patients.map((p: any) => {
            const reasonForVisit = p.symptoms || "";
            const symptoms = p.symptoms ? [p.symptoms] : [];
            
            // Use saved status/priority if available, otherwise use defaults
            const priority = p.priority || detectPriority(reasonForVisit);
            const status = p.status || "Waiting";
            
            console.log(`👤 Patient ${p.name}: status=${status}, priority=${priority}, photoPath=${p.photoPath}`);
            
            return {
                id: p.folder,
                name: p.name,
                age: 0, // Not collected yet
                gender: "Other" as const,
                dateOfBirth: "",
                phone: "",
                email: "",
                address: "",
                bloodType: "",
                insuranceProvider: "",
                insuranceNumber: "",
                emergencyContact: {
                    name: "",
                    relationship: "",
                    phone: "",
                },
                symptoms,
                reasonForVisit,
                vitalSigns: {
                    bloodPressure: "",
                    heartRate: 0,
                    temperature: 0,
                    oxygenSaturation: 0,
                    weight: 0,
                    height: 0,
                },
                medicalHistory: [],
                allergies: [],
                currentMedications: [],
                lastVisit: new Date(p.timestamp).toISOString().split('T')[0],
                status: status as Patient["status"],
                priority: priority as Patient["priority"],
                photo: p.photoPath,
            };
        });
        
        console.log('✅ Transformed patients:', transformed);
        return transformed;
    } catch (error) {
        console.error('Error loading patients:', error);
        return patients; // Fallback to mock data
    }
}

function detectPriority(text: string): "Low" | "Medium" | "High" | "Critical" {
    const lowerText = text.toLowerCase();
    
    const criticalKeywords = ['chest pain', 'heart attack', 'stroke', 'seizure', 'severe bleeding', 'unconscious'];
    const highKeywords = ['severe pain', 'difficulty breathing', 'high fever', 'broken bone', 'head injury'];
    const mediumKeywords = ['pain', 'fever', 'infection', 'injury', 'bleeding'];
    
    if (criticalKeywords.some(keyword => lowerText.includes(keyword))) {
        return 'Critical';
    }
    if (highKeywords.some(keyword => lowerText.includes(keyword))) {
        return 'High';
    }
    if (mediumKeywords.some(keyword => lowerText.includes(keyword))) {
        return 'Medium';
    }
    return 'Low';
}

export const patients: Patient[] = [
    {
        id: "P-001",
        name: "Sarah Johnson",
        age: 34,
        gender: "Female",
        dateOfBirth: "1991-03-15",
        phone: "(555) 123-4567",
        email: "sarah.johnson@email.com",
        address: "123 Oak Street, Springfield, IL 62701",
        bloodType: "A+",
        insuranceProvider: "Blue Cross Blue Shield",
        insuranceNumber: "BCBS-789456123",
        emergencyContact: {
            name: "Michael Johnson",
            relationship: "Spouse",
            phone: "(555) 987-6543",
        },
        symptoms: ["Persistent cough", "Chest tightness", "Shortness of breath", "Fatigue"],
        reasonForVisit: "Follow-up for respiratory symptoms that have persisted for 2 weeks",
        vitalSigns: {
            bloodPressure: "118/76",
            heartRate: 78,
            temperature: 98.8,
            oxygenSaturation: 97,
            weight: 145,
            height: 65,
        },
        medicalHistory: ["Asthma (childhood)", "Seasonal allergies"],
        allergies: ["Penicillin", "Shellfish"],
        currentMedications: ["Albuterol inhaler (as needed)", "Zyrtec 10mg daily"],
        lastVisit: "2025-12-28",
        status: "In Progress",
        priority: "Medium",
    },
    {
        id: "P-002",
        name: "James Martinez",
        age: 58,
        gender: "Male",
        dateOfBirth: "1967-08-22",
        phone: "(555) 234-5678",
        email: "james.m@email.com",
        address: "456 Pine Avenue, Springfield, IL 62702",
        bloodType: "O-",
        insuranceProvider: "Aetna",
        insuranceNumber: "AET-456789012",
        emergencyContact: {
            name: "Maria Martinez",
            relationship: "Wife",
            phone: "(555) 876-5432",
        },
        symptoms: ["Severe headache", "Dizziness", "Blurred vision", "Neck stiffness"],
        reasonForVisit: "Acute onset headache with neurological symptoms",
        vitalSigns: {
            bloodPressure: "158/94",
            heartRate: 88,
            temperature: 99.2,
            oxygenSaturation: 98,
            weight: 198,
            height: 70,
        },
        medicalHistory: ["Type 2 Diabetes", "Hypertension", "High cholesterol"],
        allergies: ["Sulfa drugs"],
        currentMedications: ["Metformin 1000mg twice daily", "Lisinopril 20mg daily", "Atorvastatin 40mg daily"],
        lastVisit: "2026-01-05",
        status: "Waiting",
        priority: "Critical",
    },
    {
        id: "P-003",
        name: "Emily Chen",
        age: 28,
        gender: "Female",
        dateOfBirth: "1997-11-08",
        phone: "(555) 345-6789",
        email: "emily.chen@email.com",
        address: "789 Maple Drive, Springfield, IL 62703",
        bloodType: "B+",
        insuranceProvider: "United Healthcare",
        insuranceNumber: "UHC-321654987",
        emergencyContact: {
            name: "David Chen",
            relationship: "Brother",
            phone: "(555) 765-4321",
        },
        symptoms: ["Lower back pain", "Radiating leg pain", "Numbness in left foot"],
        reasonForVisit: "Chronic lower back pain worsening over past month",
        vitalSigns: {
            bloodPressure: "112/70",
            heartRate: 72,
            temperature: 98.4,
            oxygenSaturation: 99,
            weight: 128,
            height: 63,
        },
        medicalHistory: ["Scoliosis (mild)"],
        allergies: [],
        currentMedications: ["Ibuprofen 400mg as needed"],
        lastVisit: "2025-11-15",
        status: "Waiting",
        priority: "Medium",
    },
    {
        id: "P-004",
        name: "Robert Williams",
        age: 72,
        gender: "Male",
        dateOfBirth: "1953-05-30",
        phone: "(555) 456-7890",
        email: "r.williams@email.com",
        address: "321 Elm Court, Springfield, IL 62704",
        bloodType: "AB+",
        insuranceProvider: "Medicare",
        insuranceNumber: "MED-147258369",
        emergencyContact: {
            name: "Susan Williams",
            relationship: "Daughter",
            phone: "(555) 654-3210",
        },
        symptoms: ["Chest discomfort", "Shortness of breath on exertion", "Swollen ankles"],
        reasonForVisit: "Routine cardiac follow-up with new symptoms",
        vitalSigns: {
            bloodPressure: "142/88",
            heartRate: 68,
            temperature: 97.9,
            oxygenSaturation: 94,
            weight: 185,
            height: 68,
        },
        medicalHistory: ["Coronary artery disease", "Previous MI (2019)", "Atrial fibrillation", "COPD"],
        allergies: ["Aspirin", "Contrast dye"],
        currentMedications: [
            "Warfarin 5mg daily",
            "Metoprolol 50mg twice daily",
            "Furosemide 40mg daily",
            "Spiriva inhaler",
        ],
        lastVisit: "2025-12-10",
        status: "Completed",
        priority: "High",
    },
    {
        id: "P-005",
        name: "Amanda Foster",
        age: 45,
        gender: "Female",
        dateOfBirth: "1980-09-12",
        phone: "(555) 567-8901",
        email: "a.foster@email.com",
        address: "567 Birch Lane, Springfield, IL 62705",
        bloodType: "A-",
        insuranceProvider: "Cigna",
        insuranceNumber: "CIG-963852741",
        emergencyContact: {
            name: "Thomas Foster",
            relationship: "Husband",
            phone: "(555) 543-2109",
        },
        symptoms: ["Fatigue", "Weight gain", "Cold intolerance", "Hair loss"],
        reasonForVisit: "Symptoms suggestive of thyroid dysfunction",
        vitalSigns: {
            bloodPressure: "124/80",
            heartRate: 62,
            temperature: 97.2,
            oxygenSaturation: 98,
            weight: 172,
            height: 66,
        },
        medicalHistory: ["Hypothyroidism (family history)", "Anxiety disorder"],
        allergies: ["Latex"],
        currentMedications: ["Sertraline 50mg daily"],
        lastVisit: "2025-10-20",
        status: "In Progress",
        priority: "Low",
    },
]
