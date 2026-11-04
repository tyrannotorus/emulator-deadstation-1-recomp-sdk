// The Brick has no desktop OpenGL (PowerVR: GLES and Vulkan). PSXRecomp's OpenGL renderer (gpu_gl_renderer.c) is
// linked in all the same; on the Brick the runtime draws with its software renderer and shows the frames
// through Deadstation's Vulkan presenter, so none of these is ever called: each stops the program if it is.
#include <stdio.h>
#include <stdlib.h>

static void absent(const char *name)
{
	fprintf(stderr, "deadstation: %s called, but the Brick has no OpenGL\\n", name);
	abort();
}

void glBindTexture(void) { absent("glBindTexture"); }
void glClear(void) { absent("glClear"); }
void glClearColor(void) { absent("glClearColor"); }
void glClearStencil(void) { absent("glClearStencil"); }
void glColorMask(void) { absent("glColorMask"); }
void glCopyTexSubImage2D(void) { absent("glCopyTexSubImage2D"); }
void glDeleteTextures(void) { absent("glDeleteTextures"); }
void glDisable(void) { absent("glDisable"); }
void glDrawArrays(void) { absent("glDrawArrays"); }
void glEnable(void) { absent("glEnable"); }
void glFinish(void) { absent("glFinish"); }
void glGenTextures(void) { absent("glGenTextures"); }
void glGetError(void) { absent("glGetError"); }
void glGetIntegerv(void) { absent("glGetIntegerv"); }
void glGetString(void) { absent("glGetString"); }
void glGetTexImage(void) { absent("glGetTexImage"); }
void glLineWidth(void) { absent("glLineWidth"); }
void glPixelStorei(void) { absent("glPixelStorei"); }
void glReadBuffer(void) { absent("glReadBuffer"); }
void glReadPixels(void) { absent("glReadPixels"); }
void glScissor(void) { absent("glScissor"); }
void glStencilFunc(void) { absent("glStencilFunc"); }
void glStencilMask(void) { absent("glStencilMask"); }
void glStencilOp(void) { absent("glStencilOp"); }
void glTexImage2D(void) { absent("glTexImage2D"); }
void glTexParameteri(void) { absent("glTexParameteri"); }
void glTexSubImage2D(void) { absent("glTexSubImage2D"); }
void glViewport(void) { absent("glViewport"); }
