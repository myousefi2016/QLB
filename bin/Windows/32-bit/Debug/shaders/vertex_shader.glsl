varying vec4 vColor;

void main(void)
{
   vColor = gl_Color;
   gl_Position = gl_ModelViewProjectionMatrix * gl_Vertex;
}