import React,{useEffect,useState} from "react";
import {createRoot} from "react-dom/client";
import {Car,Upload,Database,Activity} from "lucide-react";
import {BarChart,Bar,XAxis,YAxis,Tooltip,ResponsiveContainer} from "recharts";
import "./style.css";

const API=import.meta.env.VITE_API_URL||"http://localhost:8000";

function App(){
  const [file,setFile]=useState(null);
  const [result,setResult]=useState(null);
  const [slots,setSlots]=useState([]);
  const [history,setHistory]=useState([]);
  const [videoSlots,setVideoSlots]=useState([]);
  const [loading,setLoading]=useState(false);

  async function load(){
    const s=await fetch(`${API}/api/slots`).then(r=>r.json());
    setSlots(s.slots||[]);
    const h=await fetch(`${API}/api/detections`).then(r=>r.json());
    setHistory(h.detections||[]);
  }

  useEffect(()=>{load()},[]);

  async function detect(type){
    if(!file)return alert("Please select image or video");

    setLoading(true);
    setResult(null);
    setVideoSlots([]);

    const form=new FormData();
    form.append("file",file);

    try{
      const r=await fetch(`${API}/api/detect/${type}`,{
        method:"POST",body:form
      });
      const data=await r.json();
      setResult(data);
      await load();

      if(type==="video"){
        const first=data.record?.frame_summaries?.[0]?.slots;
        if(first)setVideoSlots(first);
      }
    }catch(e){
      console.error(e);
      alert("Detection failed. Check backend terminal.");
    }

    setLoading(false);
  }

  function videoUpdate(e){
    const frames=result?.record?.frame_summaries||[];
    if(!frames.length)return;

    const frame=Math.floor(e.currentTarget.currentTime*30);
    const current=frames.filter(x=>x.frame_index<=frame).at(-1);

    if(current?.slots)setVideoSlots(current.slots);
  }

  const shownSlots=videoSlots.length?videoSlots:slots;
  const occupied=shownSlots.filter(s=>s.status==="occupied").length;
  const vacant=shownSlots.filter(s=>s.status==="vacant").length;

  const chart=shownSlots.map(s=>({
    slot:s.slot_id,
    value:s.status==="occupied"?1:0
  }));

  return <div className="page">

    <h1>Smart Car Parking Detection System</h1>

    <div className="cards">
      <Card icon={<Car/>} label="Total Slots" value={shownSlots.length}/>
      <Card icon={<Activity/>} label="Occupied" value={occupied}/>
      <Card icon={<Upload/>} label="Vacant" value={vacant}/>
      <Card icon={<Database/>} label="Occupancy"
        value={shownSlots.length?Math.round(occupied/shownSlots.length*100)+"%":"0%"}/>
    </div>

    <div className="result-grid">

      <div className="panel">
        <h2>Upload Image / Video</h2>

        <input type="file" accept="image/*,video/*"
          onChange={e=>setFile(e.target.files[0])}/>

        <button onClick={()=>detect("image")}>Detect Image</button>
        <button onClick={()=>detect("video")}>Detect Video</button>

        {loading&&<p>Processing...</p>}

        <h2>Detected Result</h2>

        {result?.result_video_url ? (
          <video
            className="result"
            controls
            onTimeUpdate={videoUpdate}
            src={`${API}${result.result_video_url}`}
          />
        ) : result?.result_url ? (
          <img
            className="result"
            src={`${API}${result.result_url}`}
            alt="Detected result"
          />
        ) : null}
      </div>

      <div className="panel">
        <h2>Parking Slot Status</h2>

        <div className="slotGrid">
          {shownSlots.map(s=>
            <div className={`slot ${s.status}`} key={s.slot_id}>
              <b>{s.slot_id}</b>
              <span>{s.status}</span>
            </div>
          )}
        </div>

        <h2>Occupancy Graph</h2>

        <ResponsiveContainer width="100%" height={250}>
          <BarChart data={chart}>
            <XAxis dataKey="slot"/>
            <YAxis domain={[0,1]}/>
            <Tooltip/>
            <Bar dataKey="value"/>
          </BarChart>
        </ResponsiveContainer>
      </div>

    </div>

    <div className="panel">
      <h2>Model Evaluation (PKLot)</h2>

      <div className="cards">
        <Card label="Accuracy" value="65.38%"/>
        <Card label="Precision" value="48.03%"/>
        <Card label="Recall" value="2.44%"/>
        <Card label="F1 Score" value="4.64%"/>
        <Card label="Mean IoU" value="0.031"/>
      </div>
    </div>

    <div className="panel">
      <h2>Recent Detection History</h2>

      <table>
        <thead>
          <tr>
            <th>Source</th>
            <th>File</th>
            <th>Occupancy</th>
          </tr>
        </thead>

        <tbody>
          {history.map((h,i)=>
            <tr key={i}>
              <td>{h.source_type}</td>
              <td>{h.filename}</td>
              <td>
                {h.summary?.occupancy_rate ??
                 h.average_occupancy_rate ?? 0}%
              </td>
            </tr>
          )}
        </tbody>
      </table>
    </div>

  </div>;
}

function Card({icon,label,value}){
  return <div className="card">
    {icon}<span>{label}</span><b>{value}</b>
  </div>;
}

createRoot(document.getElementById("root")).render(<App/>);
