import React, {useEffect, useState} from 'react';
import {createRoot} from 'react-dom/client';
import {Upload, Car, Database, Activity} from 'lucide-react';
import {BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer} from 'recharts';
import './style.css';

const API = import.meta.env.VITE_API_URL || 'http://localhost:8000';

function App(){
  const [health,setHealth]=useState(null);
  const [slots,setSlots]=useState([]);
  const [history,setHistory]=useState([]);
  const [file,setFile]=useState(null);
  const [result,setResult]=useState(null);
  const [loading,setLoading]=useState(false);

  async function load(){
    const h=await fetch(`${API}/api/health`).then(r=>r.json());
    const s=await fetch(`${API}/api/slots`).then(r=>r.json());
    const d=await fetch(`${API}/api/detections`).then(r=>r.json());
    setHealth(h); setSlots(s.slots||[]); setHistory(d.detections||[]);
  }

  useEffect(()=>{load().catch(console.error)},[]);

  async function upload(kind){
    if(!file) return alert('Please select an image or video first');

    setLoading(true);
    setResult(null);

    try{
      const form=new FormData();
      form.append('file',file);

      const response=await fetch(`${API}/api/detect/${kind}`,{
        method:'POST',
        body:form
      });

      const data=await response.json();

      if(!response.ok) throw new Error(data.detail||'Detection failed');

      setResult(data);
      await load();
    }catch(error){
      alert(error.message);
    }finally{
      setLoading(false);
    }
  }

  const latest=history[0]?.summary||history[0]||{
    total_slots:slots.length,
    occupied_slots:0,
    vacant_slots:slots.length,
    occupancy_rate:0
  };

  const chartData=slots.map(s=>({
    slot:s.slot_id,
    value:s.status==='occupied'?1:0
  }));

  return <div className="page">

    <header>
      <div>
        <h1>Smart Car Parking Detection System</h1>
        <p>YOLO + OpenCV + MongoDB dashboard for parking occupancy detection</p>
      </div>
      <span className="pill">{health?.database||'loading'}</span>
    </header>

    <section className="cards">
      <Card icon={<Car/>} label="Total Slots" value={latest.total_slots??slots.length}/>
      <Card icon={<Activity/>} label="Occupied" value={latest.occupied_slots??0}/>
      <Card icon={<Upload/>} label="Vacant" value={latest.vacant_slots??0}/>
      <Card icon={<Database/>} label="Occupancy" value={`${latest.occupancy_rate??0}%`}/>
    </section>

    <main className="grid">

      <section className="panel">
        <h2>Upload Image / Video</h2>

        <input
          type="file"
          accept="image/*,video/*"
          onChange={e=>setFile(e.target.files[0])}
        />

        <div className="buttons">
          <button onClick={()=>upload('image')} disabled={loading}>
            Detect Image
          </button>

          <button onClick={()=>upload('video')} disabled={loading}>
            Detect Video
          </button>
        </div>

        {loading&&<p>Processing...</p>}

        {result?.result_url&&(
          <img
            className="result"
            src={`${API}${result.result_url}`}
            alt="Detection result"
          />
        )}

        {result?.result_video_url&&(
          <video
            className="result"
            controls
            playsInline
            src={`${API}${result.result_video_url}`}
          />
        )}
      </section>

      <section className="panel">
        <h2>Parking Slot Status</h2>

        <div className="slotGrid">
          {slots.map(s=>
            <div className={`slot ${s.status}`} key={s.slot_id}>
              <b>{s.slot_id}</b>
              <span>{s.status}</span>
            </div>
          )}
        </div>

        <ResponsiveContainer width="100%" height={220}>
          <BarChart data={chartData}>
            <XAxis dataKey="slot"/>
            <YAxis/>
            <Tooltip/>
            <Bar dataKey="value"/>
          </BarChart>
        </ResponsiveContainer>
      </section>

    </main>

    <section className="panel">
      <h2>Recent Detection History</h2>

      <table>
        <thead>
          <tr>
            <th>Source</th>
            <th>File</th>
            <th>Vehicles/Frames</th>
            <th>Occupancy</th>
          </tr>
        </thead>

        <tbody>
          {history.map((h,i)=>
            <tr key={i}>
              <td>{h.source_type}</td>
              <td>{h.filename}</td>
              <td>{h.summary?.vehicle_count??h.frames_analyzed}</td>
              <td>{h.summary?.occupancy_rate??h.average_occupancy_rate??0}%</td>
            </tr>
          )}
        </tbody>
      </table>
    </section>

  </div>
}

function Card({icon,label,value}){
  return <div className="card">
    {icon}
    <span>{label}</span>
    <b>{value}</b>
  </div>
}

createRoot(document.getElementById('root')).render(<App/>);
