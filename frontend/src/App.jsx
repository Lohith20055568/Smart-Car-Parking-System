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
const [evaluation,setEvaluation]=useState({});
const [loading,setLoading]=useState(false);

const load=async()=>{
const [s,h,e]=await Promise.all([
fetch(`${API}/api/slots`).then(r=>r.json()),
fetch(`${API}/api/detections`).then(r=>r.json()),
fetch(`${API}/api/evaluation`).then(r=>r.json())
]);
setSlots(s.slots||[]);
setHistory(h.detections||[]);
setEvaluation(e);
};

useEffect(()=>{load()},[]);

const detect=async(type)=>{
if(!file)return alert("Please select image or video");
setLoading(true);
setResult(null);
setVideoSlots([]);

const form=new FormData();
form.append("file",file);

try{
const data=await fetch(`${API}/api/detect/${type}`,{
method:"POST",
body:form
}).then(r=>r.json());

setResult(data);
await load();

if(type==="video")
setVideoSlots(data.record?.frame_summaries?.[0]?.slots||[]);

}catch(e){
console.error(e);
alert("Detection failed. Check backend terminal.");
}

setLoading(false);
};

const videoUpdate=e=>{
const frames=result?.record?.frame_summaries||[];
if(!frames.length)return;

const frame=Math.floor(e.currentTarget.currentTime*30);
const current=frames.filter(x=>x.frame_index<=frame).at(-1);

if(current?.slots)setVideoSlots(current.slots);
};
  
const shownSlots=videoSlots.length?videoSlots:slots;
const occupied=shownSlots.filter(s=>s.status==="occupied").length;
const vacant=shownSlots.filter(s=>s.status==="vacant").length;

const chart=shownSlots.map(s=>({
slot:s.slot_id,
value:s.status==="occupied"?1:0
}));

const stats=[
[<Car/>,"Total Slots",shownSlots.length],
[<Activity/>,"Occupied",occupied],
[<Upload/>,"Vacant",vacant],
[<Database/>,"Occupancy",shownSlots.length?Math.round(occupied/shownSlots.length*100)+"%":"0%"]
];

const metrics=[
["Accuracy",evaluation.Accuracy!==undefined?evaluation.Accuracy+"%":"N/A"],
["Precision",evaluation.Precision!==undefined?evaluation.Precision+"%":"N/A"],
["Recall",evaluation.Recall!==undefined?evaluation.Recall+"%":"N/A"],
["F1 Score",evaluation.F1!==undefined?evaluation.F1+"%":"N/A"],
["FPS",evaluation.FPS??"N/A"],
["Latency",evaluation.Latency_ms!==undefined?evaluation.Latency_ms+" ms":"N/A"]
];

return(
<div className="page">

<h1>Smart Car Parking Detection System</h1>

<div className="cards">
{stats.map((s,i)=><Card key={i} icon={s[0]} label={s[1]} value={s[2]}/>)}
</div>

<div className="result-grid">

<div className="panel">

<h2>Upload Image / Video</h2>

<input type="file" accept="image/*,video/*" onChange={e=>setFile(e.target.files[0])}/>

<button onClick={()=>detect("image")}>Detect Image</button>
<button onClick={()=>detect("video")}>Detect Video</button>

{loading&&<p>Processing...</p>}

<h2>Detected Result</h2>

{result?.result_video_url?
<video className="result" controls onTimeUpdate={videoUpdate} src={`${API}${result.result_video_url}`}/>
:
result?.result_url?
<img className="result" src={`${API}${result.result_url}`} alt="Detected"/>
:null}

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

<h2>Model Evaluation</h2>

<div className="cards">
{metrics.map((m,i)=>
<Card key={i} label={m[0]} value={m[1]}/>
)}
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
{h.summary?.occupancy_rate??h.average_occupancy_rate??0}%
</td>
</tr>
)}

</tbody>

</table>

</div>

</div>
);
}


function Card({icon,label,value}){
return(
<div className="card">
{icon}
<span>{label}</span>
<b>{value}</b>
</div>
);
}


createRoot(
document.getElementById("root")
).render(
<App/>
);


