import type { Patient } from "./data"

export interface Dashboard {
  activeTab: "dashboard" | "patients" | "appointments" | "records" | "notifications" | "settings"
  selectedPatient: Patient | null
}
