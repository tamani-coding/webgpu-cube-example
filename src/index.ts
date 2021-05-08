import { Cube } from './objects';
import { Scene } from './scene';
import { Camera } from './camera';
import { WebGpuRenderer } from './renderer'

const outputCanvas = document.createElement('canvas')
outputCanvas.width = window.innerWidth
outputCanvas.height = window.innerHeight
document.body.appendChild(outputCanvas)

const camera = new Camera(outputCanvas.width / outputCanvas.height);
camera.z = 10
camera.y = 10
const scene = new Scene();

const renderer = new WebGpuRenderer();
renderer.init(outputCanvas).then((success) => {
    if (!success) return;

    const cube1 = new Cube({ x: -4, y: 4 }, { r: 0.9, g: 0.01, b: 0.01 });
    const cube2 = new Cube({ y: 4 }, { r: 0.01, g: 0.9, b: 0.01 });
    const cube3 = new Cube({ x: 4, y: 4 }, { r: 0.01, g: 0.01, b: 0.9 });

    const cube4 = new Cube({ x: -4, scaleX: 0.5 }, { r: 0.5, g: 0.2, b: 0.2});
    const cube5 = new Cube({ scaleY: 0.5 }, { r: 0.2, g: 0.5, b: 0.2 });
    const cube6 = new Cube({ x: 4, scaleZ: 0.5 }, { r: 0.2, g: 0.2, b: 0.5 });

    scene.add(cube1);
    scene.add(cube2);
    scene.add(cube3);

    scene.add(cube4);
    scene.add(cube5);
    scene.add(cube6);

    const doFrame = () => {
        // ANIMATE
        const now = Date.now() / 1000;
        // for (let object of scene.getObjects()) {
        //     object.rotX = Math.sin(now)
        //     object.rotZ = Math.cos(now)
        // }
        cube1.rotX = Math.cos(now)
        cube2.rotY = Math.cos(now)
        cube3.rotZ = Math.cos(now)

        cube4.rotX = Math.sin(now)
        cube5.rotY = Math.sin(now)
        cube6.rotZ = Math.sin(now)

        // RENDER
        renderer.frame(camera, scene);
        requestAnimationFrame(doFrame);
    };
    requestAnimationFrame(doFrame);
});

window.onresize = () => {
    outputCanvas.width = window.innerWidth;
    outputCanvas.height = window.innerHeight;
    camera.aspect = outputCanvas.width / outputCanvas.height;
    renderer.update(outputCanvas);
}


function addCube() {
    scene.add(new Cube({ x: (Math.random() - 0.5) * 20, y: (Math.random() - 0.5) * 10 }));
}


// BUTTONS
const boxB = document.createElement('button')
boxB.textContent = "ADD CUBE"
boxB.classList.add('cubeButton')
boxB.onclick = addCube
document.body.appendChild(boxB)


// MOUSE CONTROLS

// ZOOM
outputCanvas.onwheel = (event: WheelEvent) => {
    camera.z += event.deltaY / 100
}

// MOUSE DRAG
var mouseDown = false;
outputCanvas.onmousedown = (event: MouseEvent) => {
    mouseDown = true;

    lastMouseX = event.pageX;
    lastMouseY = event.pageY;
}
outputCanvas.onmouseup = (event: MouseEvent) => {
    mouseDown = false;
}
var lastMouseX=-1; 
var lastMouseY=-1;
outputCanvas.onmousemove = (event: MouseEvent) => {
    if (!mouseDown) {
        return;
    }

    var mousex = event.pageX;
    var mousey = event.pageY;

    if (lastMouseX > 0 && lastMouseY > 0) {
        const roty = mousex - lastMouseX;
        const rotx = mousey - lastMouseY;

        camera.rotY += roty / 100;
        camera.rotX += rotx / 100;
    }

    lastMouseX = mousex;
    lastMouseY = mousey;
}