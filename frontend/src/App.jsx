import { useState } from 'react'
import './App.css'

const initialForm = {
  gender: 'Female',
  Partner: 'No',
  Dependents: 'No',
  tenure: 12,
  PhoneService: 'Yes',
  MultipleLines: 'No',
  InternetService: 'DSL',
  OnlineSecurity: 'No',
  OnlineBackup: 'No',
  DeviceProtection: 'No',
  TechSupport: 'No',
  StreamingTV: 'No',
  StreamingMovies: 'No',
  Contract: 'Month-to-month',
  PaperlessBilling: 'Yes',
  PaymentMethod: 'Electronic check',
  MonthlyCharges: 70,
  TotalCharges: 840,
}

function App() {
  const [formData, setFormData] = useState(initialForm)
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  const handleChange = (event) => {
    const { name, value } = event.target
    setFormData((prev) => ({ ...prev, [name]: value }))
  }

  const handleSubmit = async (event) => {
    event.preventDefault()
    setLoading(true)
    setError(null)
    setResult(null)

    try {
      const response = await fetch('http://127.0.0.1:8000/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          ...formData,
          tenure: Number(formData.tenure),
          MonthlyCharges: Number(formData.MonthlyCharges),
          TotalCharges: Number(formData.TotalCharges),
        }),
      })

      if (!response.ok) {
        throw new Error(`Request failed with status ${response.status}`)
      }

      const data = await response.json()
      setResult(data)
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="app">
      <h1>Churn Prediction</h1>
      <p className="subtitle">Enter a customer's details to check their churn risk.</p>

      <form onSubmit={handleSubmit} className="form">
        <label>
          Tenure (months)
          <input type="number" name="tenure" value={formData.tenure} onChange={handleChange} min="0" />
        </label>

        <label>
          Monthly Charges ($)
          <input type="number" name="MonthlyCharges" value={formData.MonthlyCharges} onChange={handleChange} min="0" />
        </label>

        <label>
          Total Charges ($)
          <input type="number" name="TotalCharges" value={formData.TotalCharges} onChange={handleChange} min="0" />
        </label>

        <label>
          Contract Type
          <select name="Contract" value={formData.Contract} onChange={handleChange}>
            <option>Month-to-month</option>
            <option>One year</option>
            <option>Two year</option>
          </select>
        </label>

        <label>
          Internet Service
          <select name="InternetService" value={formData.InternetService} onChange={handleChange}>
            <option>DSL</option>
            <option>Fiber optic</option>
            <option>No</option>
          </select>
        </label>

        <label>
          Payment Method
          <select name="PaymentMethod" value={formData.PaymentMethod} onChange={handleChange}>
            <option>Electronic check</option>
            <option>Mailed check</option>
            <option>Bank transfer (automatic)</option>
            <option>Credit card (automatic)</option>
          </select>
        </label>

        <button type="submit" disabled={loading}>
          {loading ? 'Predicting...' : 'Predict Churn Risk'}
        </button>
      </form>

      {error && <p className="error">Error: {error}</p>}

      {result && (
        <div className={`result ${result.churn_prediction === 1 ? 'risk' : 'stable'}`}>
          <h2>{result.churn_prediction === 1 ? 'High Risk' : 'Stable'}</h2>
          <p>Churn probability: {(result.churn_probability * 100).toFixed(1)}%</p>
        </div>
      )}
    </div>
  )
}

export default App