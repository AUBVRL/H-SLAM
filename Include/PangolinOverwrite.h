#ifndef __PangolinOverwrite__
#define __PangolinOverwrite__

#include "pangolin/display/widgets/widgets.h"
#include "pangolin/gl/gldraw.h"
#include <pangolin/display/display_internal.h>
#include <boost/thread/mutex.hpp>

namespace pangolin
{

template <typename T>
void GuiVarChanged(Var<T> &var)
{
    VarState::I().FlagVarChanged();
    var.Meta().gui_changed = true;

    for (std::vector<GuiVarChangedCallback>::iterator igvc = VarState::I().gui_var_changed_callbacks.begin(); igvc != VarState::I().gui_var_changed_callbacks.end(); ++igvc)
    {
        if (StartsWith(var.Meta().full_name, igvc->filter))
        {
            igvc->fn(igvc->data, var.Meta().full_name, var.Ref());
        }
    }
}

boost::mutex new_display_mutex;
GLfloat Transparent[4] = {0.0f, 0.0f, 0.0f, 0.0f}; //fully transparent color
GLfloat GreenTransparent[4] = {0.0f, 1.0f, 0.0f, 0.5f};
GLfloat Black[4] = {0.0f, 0.0f, 0.0f, 1.0f};
GLfloat White[4] = {1.0f, 1.0f, 1.0f, 1.0f};

GLfloat TextColor[4] = {1.0f, 1.0f, 1.0f, 1.0f};
;

void glRect(Viewport v)
{
    GLfloat vs[] = {(float)v.l, (float)v.b,
                    (float)v.l, (float)v.t(),
                    (float)v.r(), (float)v.t(),
                    (float)v.r(), (float)v.b};

    glEnableClientState(GL_VERTEX_ARRAY);
    glVertexPointer(2, GL_FLOAT, 0, vs);
    glDrawArrays(GL_TRIANGLE_FAN, 0, 4);
    glDisableClientState(GL_VERTEX_ARRAY);
}

struct PANGOLIN_EXPORT HandlerResize : Handler
{
    
    void Mouse(View& In_, MouseButton button, int x, int y, bool pressed, int button_state)
    {
        float TopBound = In_.top.p;
    
        if (button == MouseWheelDown ) //MouseButtonRight
            TopBound = std::max(0.1, TopBound - 0.01);
    
        if (button == MouseWheelUp ) //MouseButtonLeft
            TopBound = std::min(1.0, TopBound + 0.01);

        In_.SetBounds(In_.bottom, TopBound, In_.left, In_.right);
    }

    void MouseMotion(View&, int x, int y, int button_state)
    {
    }
};

struct NewCheckBox : public Checkbox
{
    NewCheckBox(std::string title, VarValueGeneric &tv) : Checkbox(title, tv) {}

    void Render()
    {
        const bool val = var->Get();

        if (val)
        {
            glColor4fv(GreenTransparent); //when activated
            glRect(vcb);
        }
        glColor4fv(TextColor);
        gltext.DrawWindow(raster[0], raster[1]);
        // DrawShadowRect(vcb, val);
    }
};

struct NewButton : public Button
{   //KNOWN BUG: this button was created for the recorder URI, when the window is resized the recorder stops working.
    //Since there is no feedback from the recorder the button assumes is still on. Tried detecting window resizing 
    //but did not work!
    bool IsOn = false;
    GlText textIfOn;
    GlText textIfOff;
    bool ChangeableName = false;
    NewButton(std::string title, VarValueGeneric &tv) : Button(title, tv)
    {
        if (title.find('!') != std::string::npos) //ex: Record Video!Stop Recording
        {
            gltext = textIfOff = GlFont::I().Text(title.substr(0, title.find('!'))); //Record Video
            textIfOn = GlFont::I().Text(title.substr(title.find('!') + 1));          //Stop Recording
            ChangeableName = true;
        }
        else
            gltext = textIfOff = textIfOn = GlFont::I().Text(title);
    }

    void Render()
    {
        glColor4fv(Transparent);
        glRect(v);
        glColor4fv(TextColor);
        if (ChangeableName)
            if (IsOn)
                textIfOn.DrawWindow(raster[0], raster[1] - down);
            else
                textIfOff.DrawWindow(raster[0], raster[1] - down);
        else
            gltext.DrawWindow(raster[0], raster[1] - down);
        // DrawShadowRect(v, down);
    }

    void Mouse(View &, MouseButton button, int /*x*/, int /*y*/, bool pressed, int /*mouse_state*/)
    {
        if (button == MouseButtonLeft)
        {
            down = pressed;
            if (!pressed)
            {
                IsOn = !IsOn;
                var->Set(!var->Get());
                GuiVarChanged(*this);
            }
        }
    }
};

struct NewPanel : public Panel
{
    void Render()
    {
#ifndef HAVE_GLES
        glPushAttrib(GL_CURRENT_BIT | GL_ENABLE_BIT | GL_DEPTH_BUFFER_BIT | GL_SCISSOR_BIT | GL_VIEWPORT_BIT | GL_COLOR_BUFFER_BIT | GL_TRANSFORM_BIT);
#endif
        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

        DisplayBase().ActivatePixelOrthographic();
        glDisable(GL_DEPTH_TEST);
        glDisable(GL_LIGHTING);
        glDisable(GL_SCISSOR_TEST);
        glDisable(GL_LINE_SMOOTH);
        glDisable(GL_COLOR_MATERIAL);
        glLineWidth(1.0);

        glColor4fv(Transparent);

        RenderChildren();

#ifndef HAVE_GLES
        glPopAttrib();
#else
        glEnable(GL_LINE_SMOOTH);
        glEnable(GL_DEPTH_TEST);
#endif
    }

    static void AddNewVariable(void *data, const std::string &name, VarValueGeneric &var, bool /*brand_new*/)
    {
        NewPanel *thisptr = (NewPanel *)data;

        const std::string &title = var.Meta().friendly;

        new_display_mutex.lock();

        ViewMap::iterator pnl = GetCurrentContext()->named_managed_views.find(name);

        if (pnl == GetCurrentContext()->named_managed_views.end())
        {
            View *nv = NULL;
            if (!strcmp(var.TypeId(), typeid(bool).name()))
            {
                nv = (var.Meta().flags & META_FLAG_TOGGLE) ? (View *)new NewCheckBox(title, var) : (View *)new NewButton(title, var);
            }
            else if (!strcmp(var.TypeId(), typeid(double).name()) ||
                     !strcmp(var.TypeId(), typeid(float).name()) ||
                     !strcmp(var.TypeId(), typeid(int).name()) ||
                     !strcmp(var.TypeId(), typeid(unsigned int).name()))
            {
                nv = new Slider(title, var);
            }
            else if (!strcmp(var.TypeId(), typeid(std::function<void(void)>).name()))
            {
                nv = (View *)new FunctionButton(title, var);
            }
            else
            {
                nv = new TextInput(title, var);
            }
            if (nv)
            {
                GetCurrentContext()->named_managed_views[name] = nv;
                thisptr->views.push_back(nv);
                thisptr->ResizeChildren();
            }
        }

        new_display_mutex.unlock();
    }

    NewPanel(const std::string &auto_register_var_prefix) : Panel()
    {
        pangolin::RegisterNewVarCallback(&AddNewVariable, (void *)this, auto_register_var_prefix);
        pangolin::ProcessHistoricCallbacks(&AddNewVariable, (void *)this, auto_register_var_prefix);
    }
};

PANGOLIN_EXPORT View &CreateNewPanel(const std::string &name)
{
    if (GetCurrentContext()->named_managed_views.find(name) != GetCurrentContext()->named_managed_views.end())
    {
        throw std::runtime_error("Panel already registered with this name.");
    }
    NewPanel *p = new NewPanel(name);
    GetCurrentContext()->named_managed_views[name] = p;
    GetCurrentContext()->base.views.push_back(p);
    return *p;
}
} // namespace pangolin

#endif