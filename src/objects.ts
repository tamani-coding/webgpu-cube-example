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
                transform : mat4x4<f32>;    // translate AND rotate
                rotate : mat4x4<f32>;       // rotate only
            };

            [[block]] struct Camera {     // 4x4 transform matrix
                matrix : mat4x4<f32>;
            };

            [[block]] struct Color {        // RGB color
                color: vec3<f32>;
            };
            
            // bind model/camera/color buffers
            [[group(0), binding(0)]] var<uniform> modelTransform    : Uniforms;
            [[group(0), binding(2)]] var<uniform> cameraTransform   : Camera;
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
                var transformedPosition: vec4<f32> = modelTransform.transform * vec4<f32>(input.position, 1.0);

                output.Position = cameraTransform.matrix * transformedPosition;             // transformed with model & camera projection
                output.fragColor = color.color;                                             // fragment color from buffer
                output.fragNorm = (modelTransform.rotate * vec4<f32>(input.norm, 1.0)).xyz; // transformed normal vector with model
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
            `
            + bindSamplerAndTexture +
            `
            [[stage(fragment)]]
            fn main(input : FragmentInput) -> [[location(0)]] vec4<f32> {
                let lightDirection: vec3<f32> = normalize(lightData.lightPos - input.fragPos);

                // lambert factor
                let lambertFactor : f32 = dot(lightDirection, input.fragNorm);

                var lightFactor: f32 = 0.0;
                lightFactor = lambertFactor;

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
    private offset = 256; // transformationBindGroup offset must be 256-byte aligned
    private uniformBufferSize = this.offset + 2 * this.matrixSize;

    private transformMatrix = mat4.create() as Float32Array;
    private rotateMatrix = mat4.create() as Float32Array;

    private renderPipeline: GPURenderPipeline;
    private transformationBuffer: GPUBuffer;
    private transformationBindGroup: GPUBindGroup;
    private verticesBuffer: GPUBuffer;
    private colorBuffer: GPUBuffer;

    private perVertex = ( 3 + 3 + 2 );      // 3 for position, 3 for normal, 2 for uv, 3 for color
    private stride = this.perVertex * 4;    // stride = byte length of vertex data array 

    constructor(parameter?: CubeParameter, color?: Color, imageBitmap?: ImageBitmap) {
        this.setTransformation(parameter);
        this.renderPipeline = device.createRenderPipeline({
            vertex: {
                module: device.createShaderModule({ code: vertxShader(),}),
                entryPoint: 'main',
                buffers: [
                    {
                        arrayStride: this.stride, // ( 3 (pos) + 3 (norm) + 2 (uv) ) * 4 bytes
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
                module: device.createShaderModule({ code: fragmentShader(imageBitmap != null), }),
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

        this.transformationBuffer = device.createBuffer({
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
                    buffer: this.transformationBuffer,
                    offset: 0,
                    size: this.matrixSize * 2,
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

        this.transformationBindGroup = device.createBindGroup({
            layout: this.renderPipeline.getBindGroupLayout(0),
            entries: entries as Iterable<GPUBindGroupEntry>,
        });
    }

    public draw(passEncoder: GPURenderPassEncoder, device: GPUDevice) {
        this.updateTransformationMatrix()

        passEncoder.setPipeline(this.renderPipeline);
        device.queue.writeBuffer(
            this.transformationBuffer,
            0,
            this.transformMatrix.buffer,
            this.transformMatrix.byteOffset,
            this.transformMatrix.byteLength
        );
        device.queue.writeBuffer(
            this.transformationBuffer,
            64,
            this.rotateMatrix.buffer,
            this.rotateMatrix.byteOffset,
            this.rotateMatrix.byteLength
        );
        passEncoder.setVertexBuffer(0, this.verticesBuffer);
        passEncoder.setBindGroup(0, this.transformationBindGroup);
        passEncoder.draw(vertices.length, 1, 0, 0);
    }

    private updateTransformationMatrix() {
        // MOVE / TRANSLATE OBJECT
        const transform = mat4.create();
        const rotate = mat4.create();

        mat4.translate(transform, transform, vec3.fromValues(this.x, this.y, this.z))
        mat4.rotateX(transform, transform, this.rotX);
        mat4.rotateY(transform, transform, this.rotY);
        mat4.rotateZ(transform, transform, this.rotZ);

        mat4.rotateX(rotate, rotate, this.rotX);
        mat4.rotateY(rotate, rotate, this.rotY);
        mat4.rotateZ(rotate, rotate, this.rotZ);

        // APPLY
        mat4.copy(this.transformMatrix, transform)
        mat4.copy(this.rotateMatrix, rotate)
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
