import os,json,time,cv2,matplotlib.pyplot as plt
from sklearn.metrics import *
from .detection import detect_vehicles,calculate_iou


ROOT=os.path.abspath(os.path.join(os.path.dirname(__file__),"../.."))
DATA=os.path.join(ROOT,"datasets/PKLot")

ANN=os.path.join(DATA,"annotations.json")
IMG=os.path.join(DATA,"images copy")


def inside(slot,v):
    cx=(v["x1"]+v["x2"])/2
    cy=(v["y1"]+v["y2"])/2
    return slot["x1"]<=cx<=slot["x2"] and slot["y1"]<=cy<=slot["y2"]


def evaluate():

    data=json.load(open(ANN))

    cats={c["id"]:c["name"] for c in data["categories"]}

    imgs={i["id"]:i["file_name"] for i in data["images"]}


    y,yh,ious,t=[],[],[],[]


    for iid,name in imgs.items():

        frame=cv2.imread(os.path.join(IMG,name))

        if frame is None:
            continue


        s=time.time()
        det=detect_vehicles(frame)
        t.append(time.time()-s)


        for a in data["annotations"]:

            if a["image_id"]!=iid:
                continue

            if cats[a["category_id"]] not in ["space-empty","space-occupied"]:
                continue


            slot={
            "x1":a["bbox"][0],
            "y1":a["bbox"][1],
            "x2":a["bbox"][0]+a["bbox"][2],
            "y2":a["bbox"][1]+a["bbox"][3]
            }


            occ=False
            mx=0


            for d in det:

                occ |= inside(slot,d)

                mx=max(mx,calculate_iou(slot,d))


            ious.append(mx)

            y.append(
                cats[a["category_id"]]=="space-occupied"
            )

            yh.append(occ)



    cm=confusion_matrix(y,yh)
    tn,fp,fn,tp=cm.ravel()


    result={
    "TP":tp,
    "TN":tn,
    "FP":fp,
    "FN":fn,
    "Accuracy":round(accuracy_score(y,yh)*100,2),
    "Precision":round(precision_score(y,yh,zero_division=0)*100,2),
    "Recall":round(recall_score(y,yh,zero_division=0)*100,2),
    "F1":round(f1_score(y,yh,zero_division=0)*100,2),
    "IoU":round(sum(ious)/len(ious),3),
    "Latency_ms":round(sum(t)/len(t)*1000,2),
    "FPS":round(1/(sum(t)/len(t)),2)
    }


    print(json.dumps(result,indent=2))


    plt.imshow(cm,cmap="Blues")
    plt.title("Parking Occupancy Confusion Matrix")
    plt.savefig("confusion_matrix.png")


if __name__=="__main__":
    evaluate()
