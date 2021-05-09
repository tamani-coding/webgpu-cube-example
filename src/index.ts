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

    // unscaled cubes
    const cube1 = new Cube({ x: -4, y: 4 }, { r: 0.9, g: 0.01, b: 0.01 });
    const cube2 = new Cube({ y: 4 }, { r: 0.01, g: 0.9, b: 0.01 });
    const cube3 = new Cube({ x: 4, y: 4 }, { r: 0.01, g: 0.01, b: 0.9 });

    scene.add(cube1);
    scene.add(cube2);
    scene.add(cube3);

    // textured cubes
    const texturedCubes: Cube[] = [];

    const texture1 = document.createElement('img');
    texture1.src = './crocodile_gena.png';
    texture1.decode().then( () => {
    
        createImageBitmap(texture1).then( (imageBitmap: ImageBitmap) => {
            const cube = new Cube({ x: -4 , scaleY: 0.7 }, null, imageBitmap);
            texturedCubes.push(cube);
            scene.add(cube);
        });
    
    });

    const texture2 = document.createElement('img');
    texture2.src = './terranigma.png';
    texture2.decode().then( () => {
    
        createImageBitmap(texture2).then( (imageBitmap: ImageBitmap) => {
            const cube = new Cube({ }, null, imageBitmap);
            texturedCubes.push(cube);
            scene.add(cube);
        });
    
    });

    const texture3 = document.createElement('img');
    texture3.src = './deno.png';
    texture3.decode().then( () => {
    
        createImageBitmap(texture3).then( (imageBitmap: ImageBitmap) => {
            const cube = new Cube({ x: 4, scaleZ: 0.8}, null, imageBitmap);
            texturedCubes.push(cube);
            scene.add(cube);
        });
    
    });

    // scaled cubes
    const cube4 = new Cube({ x: -4, y: -4, scaleX: 0.8 }, { r: 1.0, g: 1.0, b: 0.2});
    const cube5 = new Cube({ y: -4, scaleY: 0.8 }, { r: 0.2, g: 1.0, b: 1.0 });
    const cube6 = new Cube({ x: 4, y: -4, scaleZ: 0.8 }, { r: 1.0, g: 0.2, b: 1.0 });

    scene.add(cube4);
    scene.add(cube5);
    scene.add(cube6);

    const lightDebugCube = new Cube({ scaleX: 0.1, scaleY: 0.1, scaleZ: 0.1 },{r: 1.0, g: 1.0, b: 0.0});
    lightDebugCube.rotX = Math.PI / 4;
    lightDebugCube.rotZ = Math.PI / 4;
    scene.add(lightDebugCube)

    const doFrame = () => {
        // ANIMATE
        const now = Date.now() / 1000;

        cube1.rotX = Math.cos(now)
        cube2.rotY = Math.cos(now)
        cube3.rotZ = Math.cos(now)

        cube4.rotX = Math.sin(now)
        cube5.rotY = Math.sin(now)
        cube5.rotX = 0.25
        cube6.rotZ = Math.sin(now)

        for (let c of texturedCubes) {
            c.rotX = Math.cos(now);
            c.rotY = Math.sin(now);
        }

        // MOVE LIGHT AND LIGHT DEBUG CUBE
        scene.pointLightPosition[0] = Math.cos(now) * 4;
        scene.pointLightPosition[1] = Math.sin(now) * 4;
        scene.pointLightPosition[2] = 2;
        lightDebugCube.x = scene.pointLightPosition[0]
        lightDebugCube.y = scene.pointLightPosition[1]
        lightDebugCube.z = scene.pointLightPosition[2]

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

// MOUSE CONTROLS

// ZOOM
outputCanvas.onwheel = (event: WheelEvent) => {
    const delta = event.deltaY / 100;
    // no negative camera.z
    if(camera.z > -delta) {
        camera.z += event.deltaY / 100
    }
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