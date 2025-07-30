using Luxor

function bell_curve_points(; width = 60.0, height = 60.0, step = 1.5)
    pts = Point[]
    for x in -width:step:width
        y = -exp(-x^2 / (2 * (width / 2)^2)) * height
        push!(pts, Point(x, y))
    end
    return pts
end

function draw_inla_logo_colored(; filename = "../src/assets/logo.svg", bottom = -18.0, linewidth = 3.0)
    Drawing(170, 170, filename)
    origin(Point(85.0, 155.0))

    setline(linewidth)

    # Julia color scheme (outer to inner)
    julia_colors = ["#9558B2", "#CB3C33", "#389826", "#4063D8"]

    prototype = bell_curve_points(width = 60.0, height = 100.0, step = 1.0)
    N = length(julia_colors)

    for i in 1:N
        scale_factor = 1.0 + 0.1 * (i - 1)
        sethue(julia_colors[i])
        scaled_pts = [p * scale_factor for p in prototype]
        poly(scaled_pts, :stroke, close = false)
    end

    # Central filled curve (smallest, same as last in loop)
    sethue(julia_colors[1])
    #bottom = [Point(p.x, bottom) for p in reverse(prototype)]

    # Adjusted bottom curve - raised and slightly curved upwards
    bottom_pts = Point[]
    for p in reverse(prototype)
        # Raise the bottom baseline with a gentle parabolic curve
        base_y = bottom + 0.0015 * p.x^2  # slight upward curve
        push!(bottom_pts, Point(p.x, base_y))
    end

    poly(vcat(prototype, bottom_pts), :fill)

    fontface("HelveticaBold")
    fontsize(40)
    sethue("#FFFFFF")  # deep blue to match color scheme
    text("∑", Point(1, -55), halign = :center, valign = :middle)

    finish()
    return preview()
end

draw_inla_logo_colored()
