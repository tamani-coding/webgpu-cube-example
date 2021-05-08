import { device, cameraUniformBuffer, lightDataBuffer } from './renderer';
import { mat4, vec3 } from 'gl-matrix';
import { lightDataSize } from './scene';


const vertices = [
    // front
    { pos: [-1, -1,  1], norm: [ 0,  0,  1], uv: [0, 0], },
    { pos: [ 1, -1,  1], norm: [ 0,  0,  1], uv: [1, 0], },
    { pos: [-1,  1,  1], norm: [ 0,  0,  1], uv: [0, 1], },
   
    { pos: [-1,  1,  1], norm: [ 0,  0,  1], uv: [0, 1], },
    { pos: [ 1, -1,  1], norm: [ 0,  0,  1], uv: [1, 0], },
    { pos: [ 1,  1,  1], norm: [ 0,  0,  1], uv: [1, 1], },
    // right
    { pos: [ 1, -1,  1], norm: [ 1,  0,  0], uv: [0, 0], },
    { pos: [ 1, -1, -1], norm: [ 1,  0,  0], uv: [1, 0], },
    { pos: [ 1,  1,  1], norm: [ 1,  0,  0], uv: [0, 1], },
   
    { pos: [ 1,  1,  1], norm: [ 1,  0,  0], uv: [0, 1], },
    { pos: [ 1, -1, -1], norm: [ 1,  0,  0], uv: [1, 0], },
    { pos: [ 1,  1, -1], norm: [ 1,  0,  0], uv: [1, 1], },
    // back
    { pos: [ 1, -1, -1], norm: [ 0,  0, -1], uv: [0, 0], },
    { pos: [-1, -1, -1], norm: [ 0,  0, -1], uv: [1, 0], },
    { pos: [ 1,  1, -1], norm: [ 0,  0, -1], uv: [0, 1], },
   
    { pos: [ 1,  1, -1], norm: [ 0,  0, -1], uv: [0, 1], },
    { pos: [-1, -1, -1], norm: [ 0,  0, -1], uv: [1, 0], },
    { pos: [-1,  1, -1], norm: [ 0,  0, -1], uv: [1, 1], },
    // left
    { pos: [-1, -1, -1], norm: [-1,  0,  0], uv: [0, 0], },
    { pos: [-1, -1,  1], norm: [-1,  0,  0], uv: [1, 0], },
    { pos: [-1,  1, -1], norm: [-1,  0,  0], uv: [0, 1], },
   
    { pos: [-1,  1, -1], norm: [-1,  0,  0], uv: [0, 1], },
    { pos: [-1, -1,  1], norm: [-1,  0,  0], uv: [1, 0], },
    { pos: [-1,  1,  1], norm: [-1,  0,  0], uv: [1, 1], },
    // top
    { pos: [ 1,  1, -1], norm: [ 0,  1,  0], uv: [0, 0], },
    { pos: [-1,  1, -1], norm: [ 0,  1,  0], uv: [1, 0], },
    { pos: [ 1,  1,  1], norm: [ 0,  1,  0], uv: [0, 1], },
   
    { pos: [ 1,  1,  1], norm: [ 0,  1,  0], uv: [0, 1], },
    { pos: [-1,  1, -1], norm: [ 0,  1,  0], uv: [1, 0], },
    { pos: [-1,  1,  1], norm: [ 0,  1,  0], uv: [1, 1], },
    // bottom
    { pos: [ 1, -1,  1], norm: [ 0, -1,  0], uv: [0, 0], },
    { pos: [-1, -1,  1], norm: [ 0, -1,  0], uv: [1, 0], },
    { pos: [ 1, -1, -1], norm: [ 0, -1,  0], uv: [0, 1], },
   
    { pos: [ 1, -1, -1], norm: [ 0, -1,  0], uv: [0, 1], },
    { pos: [-1, -1,  1], norm: [ 0, -1,  0], uv: [1, 0], },
    { pos: [-1, -1, -1], norm: [ 0, -1,  0], uv: [1, 1], },
  ];

/** 
 * 
 * This shader calculates and outputs position and normal vector of current fragment,
 * also outputs fragment color and uv.
 * The result is piped to fragment shader
 * 
 * */ 
function vertxShader(): string {
    return `
            [[block]] struct Uniforms {     // 4x4 transform matrices
                matrix : mat4x4<f32>;
            };

            [[block]] struct Color {        // RGB color
                color: vec3<f32>;
            };
            
            // bind model/camera/color buffers
            [[group(0), binding(0)]] var<uniform> modelTransform    : Uniforms;
            [[group(0), binding(2)]] var<uniform> cameraTransform   : Uniforms;
            [[group(0), binding(1)]] var<storage> color             : [[access(read)]]  Color;
            
            // output struct of this vertex shader
            struct VertexOutput {
                [[builtin(position)]] Position : vec4<f32>;

                [[location(0)]] fragColor : vec3<f32>;
                [[location(1)]] fragNorm : vec3<f32>;
                [[location(2)]] uv : vec2<f32>;
                [[location(3)]] fragPos : vec3<f32>;
            };

            // input struct according to vertex buffer stride
            struct VertexInput {
                [[location(0)]] position : vec3<f32>;
                [[location(1)]] norm : vec3<f32>;
                [[location(2)]] uv : vec2<f32>;
            };
            
            [[stage(vertex)]]
            fn main(input: VertexInput) -> VertexOutput {
                var output: VertexOutput;
                var transformedPosition: vec4<f32> = modelTransform.matrix * vec4<f32>(input.position, 1.0);

                output.Position = cameraTransform.matrix * transformedPosition;             // transformed with model & camera projection
                output.fragColor = color.color;                                             // fragment color from buffer
                output.fragNorm = (modelTransform.matrix * vec4<f32>(input.norm, 1.0)).xyz; // transformed normal vector with model
                output.uv = input.uv;                                                       // transformed uv
                output.fragPos = transformedPosition.xyz;                                   // transformed fragment position with model

                return output;
            }
        `;
}

/**
 * This shader receives the output of the vertex shader program.
 * If texture is set, the sampler and texture is binded to this shader.
 * Determines the color of the current fragment, takes into account point light.
 * 
 */
function fragmentShader(withTexture: boolean): string {
    // conditionally bind sampler and texture, only if texture is set
    const bindSamplerAndTexture = withTexture ? `
                [[group(0), binding(4)]] var mySampler: sampler;
                [[group(0), binding(5)]] var myTexture: texture_2d<f32>;
            ` : ``;

    // conditionally do texture sampling
    const returnStatement = withTexture ? `
                                return vec4<f32>(textureSample(myTexture, mySampler, input.uv).xyz * lightingFactor, 1.0);
                            ` : `
                                return vec4<f32>(input.fragColor  * lightingFactor, 1.0);
                            `;

    return  `
            [[block]] struct LightData {        // light xyz position
                lightPos : vec3<f32>;
            };

            struct FragmentInput {              // output from vertex stage shader
                [[location(0)]] fragColor : vec3<f32>;
                [[location(1)]] fragNorm : vec3<f32>;
                [[location(2)]] uv : vec2<f32>;
                [[location(3)]] fragPos : vec3<f32>;
            };

            // bind light data buffer
            [[group(0), binding(3)]] var<uniform> lightData : LightData;

            // constants for light
            let ambientLightFactor : f32 = 0.25;     // ambient light
            let lightRange: f32 = 4.0;             // point light range
            let PI: f32 = 3.14159265359;            // PI constant
            `
            + bindSamplerAndTexture +
            `
            [[stage(fragment)]]
            fn main(input : FragmentInput) -> [[location(0)]] vec4<f32> {
                let lightDirection: vec3<f32> = normalize(lightData.lightPos - input.fragPos);
                let fnormal: vec3<f32> = vec3<f32>(0.0,0.0,-1.0);

                // calculate angle between light direction and fragment normal 
                let dot: f32 = dot(input.fragNorm, lightDirection);
                let normA: f32 = sqrt( pow(lightDirection.x, 2.0) + pow(lightDirection.y, 2.0) + pow(lightDirection.z, 2.0));
                let normB: f32 = sqrt( pow(input.fragNorm.x, 2.0) + pow(input.fragNorm.y, 2.0) + pow(input.fragNorm.z, 2.0));
                let angle: f32 = acos( dot / (normA * normB) );

                // lambert factor
                let lambertFactor : f32 = dot(lightDirection, input.fragNorm);

                let th1: f32 = PI / 2.0;
                let th2: f32 = PI + PI / 2.0;

                var lightFactor: f32 = 0.0;
                // if (angle < th1 || angle > th2) {
                    lightFactor = lambertFactor;
                // }

                let lightingFactor: f32 = max(min(lightFactor, 1.0), ambientLightFactor);
        ` + 
                returnStatement +
        `
            }
        `;
}

export interface CubeParameter {

    x?: number;
    y?: number;
    z?: number;

    rotX?: number;
    rotY?: number;
    rotZ?: number;

    scaleX?: number;
    scaleY?: number;
    scaleZ?: number;
}

export interface Color {

    r: number;
    g: number;
    b: number;

}

export class Cube {

    public x: number = 0;
    public y: number = 0;
    public z: number = 0;

    public rotX: number = 0;
    public rotY: number = 0;
    public rotZ: number = 0;

    public scaleX: number = 1;
    public scaleY: number = 1;
    public scaleZ: number = 1;

    private defaultColor: Color = {
        r: 0.9,
        g: 0.6,
        b: 0.1,
    }

    private matrixSize = 4 * 16; // 4x4 matrix
    private offset = 256; // uniformBindGroup offset must be 256-byte aligned
    private uniformBufferSize = this.offset + this.matrixSize;

    private modelViewProjectionMatrix = mat4.create() as Float32Array;

    private renderPipeline: GPURenderPipeline;
    private uniformBuffer: GPUBuffer;
    private uniformBindGroup: GPUBindGroup;
    private verticesBuffer: GPUBuffer;
    private colorBuffer: GPUBuffer;

    private perVertex = ( 3 + 3 + 2 );      // 3 for position, 3 for normal, 2 for uv, 3 for color
    private stride = this.perVertex * 4;    // stride = byte length of vertex data array 

    constructor(parameter?: CubeParameter, color?: Color, imageBitmap?: ImageBitmap) {
        this.setTransformation(parameter);

        this.renderPipeline = device.createRenderPipeline({
            vertex: {
                module: device.createShaderModule({
                    code: vertxShader(),
                }),
                entryPoint: 'main',
                buffers: [
                    {
                        arrayStride: this.stride,
                        attributes: [
                            {
                                // position
                                shaderLocation: 0,
                                offset: 0,
                                format: 'float32x3',
                            },
                            {
                                // norm
                                shaderLocation: 1,
                                offset: 3 * 4,
                                format: 'float32x3',
                            },
                            {
                                // uv
                                shaderLocation: 2,
                                offset: (3 + 3) * 4,
                                format: 'float32x2',
                            },
                        ],
                    } as GPUVertexBufferLayout,
                ],
            },
            fragment: {
                module: device.createShaderModule({
                    code: fragmentShader(imageBitmap != null),
                }),
                entryPoint: 'main',
                targets: [
                    {
                        format: 'bgra8unorm' as GPUTextureFormat,
                    },
                ],
            },
            primitive: {
                topology: 'triangle-list',
                cullMode: 'back',
            },
            // Enable depth testing so that the fragment closest to the camera
            // is rendered in front.
            depthStencil: {
                depthWriteEnabled: true,
                depthCompare: 'less',
                format: 'depth24plus-stencil8',
            },
        });

        this.uniformBuffer = device.createBuffer({
            size: this.uniformBufferSize,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

        this.colorBuffer = device.createBuffer({
            mappedAtCreation: true,
            size: Float32Array.BYTES_PER_ELEMENT * 3,
            usage: GPUBufferUsage.STORAGE,
        });
        const colorMapping = new Float32Array(this.colorBuffer.getMappedRange());
        colorMapping.set(color ? [color.r, color.g, color.b] : [this.defaultColor.r, this.defaultColor.g, this.defaultColor.b], 0);
        this.colorBuffer.unmap()

        const entries = [
            {
                binding: 0,
                resource: {
                    buffer: this.uniformBuffer,
                    offset: 0,
                    size: this.matrixSize,
                },
            },
            {
                binding: 1,
                resource: {
                    buffer: this.colorBuffer ,
                    offset: 0,
                    size: Float32Array.BYTES_PER_ELEMENT * 3,
                },
            },
            {
                binding: 2,
                resource: {
                    buffer: cameraUniformBuffer,
                    offset: 0,
                    size: this.matrixSize,
                },
            },
            {
                binding: 3,
                resource: {
                    buffer: lightDataBuffer,
                    offset: 0,
                    size: lightDataSize,
                },
            },
            
        ];

        // Texture
        if (imageBitmap) {
            let cubeTexture = device.createTexture({
                size: [imageBitmap.width, imageBitmap.height, 1],
                format: 'rgba8unorm',
                usage: GPUTextureUsage.SAMPLED | GPUTextureUsage.COPY_DST,
            });
            device.queue.copyImageBitmapToTexture(
                { imageBitmap },
                { texture: cubeTexture },
                [imageBitmap.width, imageBitmap.height, 1]
            );
            const sampler = device.createSampler({
                magFilter: 'linear',
                minFilter: 'linear',
            });

            entries.push({
                binding: 4,
                resource: sampler,
            } as any)
            entries.push({
                binding: 5,
                resource: cubeTexture.createView(),
            } as any);

        }

        this.uniformBindGroup = device.createBindGroup({
            layout: this.renderPipeline.getBindGroupLayout(0),
            entries: entries as Iterable<GPUBindGroupEntry>,
        });

        this.verticesBuffer = device.createBuffer({
            size: vertices.length * this.stride,
            usage: GPUBufferUsage.VERTEX,
            mappedAtCreation: true,
        });

        const mapping = new Float32Array(this.verticesBuffer.getMappedRange());
        for (let i = 0; i < vertices.length; i++) {
            // (3 * 4) + (3 * 4) + (2 * 4)
            mapping.set([vertices[i].pos[0] * this.scaleX, 
                        vertices[i].pos[1] * this.scaleY, 
                        vertices[i].pos[2] * this.scaleZ], this.perVertex * i + 0);
            mapping.set(vertices[i].norm, this.perVertex * i + 3);
            mapping.set(vertices[i].uv, this.perVertex * i + 6);
        }
        this.verticesBuffer.unmap();
    }

    public draw(passEncoder: GPURenderPassEncoder, device: GPUDevice) {
        this.updateTransformationMatrix()

        passEncoder.setPipeline(this.renderPipeline);
        device.queue.writeBuffer(
            this.uniformBuffer,
            0,
            this.modelViewProjectionMatrix.buffer,
            this.modelViewProjectionMatrix.byteOffset,
            this.modelViewProjectionMatrix.byteLength
        );
        passEncoder.setVertexBuffer(0, this.verticesBuffer);
        passEncoder.setBindGroup(0, this.uniformBindGroup);
        passEncoder.draw(vertices.length, 1, 0, 0);
    }

    private updateTransformationMatrix() {
        // MOVE / TRANSLATE OBJECT
        const modelMatrix = mat4.create();
        mat4.translate(modelMatrix, modelMatrix, vec3.fromValues(this.x, this.y, this.z))
        mat4.rotateX(modelMatrix, modelMatrix, this.rotX);
        mat4.rotateY(modelMatrix, modelMatrix, this.rotY);
        mat4.rotateZ(modelMatrix, modelMatrix, this.rotZ);

        // APPLY
        mat4.copy(this.modelViewProjectionMatrix, modelMatrix)
    }

    private setTransformation(parameter?: CubeParameter) {
        if (parameter == null) {
            return;
        }

        this.x = parameter.x ? parameter.x : 0;
        this.y = parameter.y ? parameter.y : 0;
        this.z = parameter.z ? parameter.z : 0;

        this.rotX = parameter.rotX ? parameter.rotX : 0;
        this.rotY = parameter.rotY ? parameter.rotY : 0;
        this.rotZ = parameter.rotZ ? parameter.rotZ : 0;

        this.scaleX = parameter.scaleX ? parameter.scaleX : 1;
        this.scaleY = parameter.scaleY ? parameter.scaleY : 1;
        this.scaleZ = parameter.scaleZ ? parameter.scaleZ : 1;
    }
}
